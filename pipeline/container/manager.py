import hashlib
import importlib
import logging
import typing as t
import urllib.parse
import zipfile
from pathlib import Path
from types import NoneType, UnionType
from urllib import request

import validators

from pipeline.cloud.schemas import runs as run_schemas
from pipeline.objects import Directory, File, Graph
from pipeline.objects.graph import InputField, InputSchema

logger = logging.getLogger("uvicorn")


class Manager:
    def __init__(self, pipeline_path: str):
        if ":" not in pipeline_path:
            raise ValueError(
                "Invalid pipeline path, must be in format <module_or_file>:<pipeline>"
            )
        if len(pipeline_path.split(":")) != 2:
            raise ValueError(
                "Invalid pipeline path, must be in format <module_or_file>:<pipeline>"
            )

        self.pipeline_path = pipeline_path
        self.pipeline_module_str, self.pipeline_name_str = pipeline_path.split(":")

        try:
            self.pipeline_module = importlib.import_module(self.pipeline_module_str)
            self.pipeline = getattr(self.pipeline_module, self.pipeline_name_str)
        except ModuleNotFoundError:
            raise ValueError(f"Could not find module {self.pipeline_module_str}")
        except AttributeError:
            raise ValueError(
                f"Could not find pipeline {self.pipeline_name_str} in module {self.pipeline_module_str}"
            )
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")

        logger.info(f"Pipeline set to {self.pipeline_path}")

    async def startup(self):
        logger.info("Starting pipeline")
        self.pipeline._startup()
        logger.info("Pipeline started successfully")

    def _resolve_file_variable_to_local(
        self,
        file: File | Directory,
        *,
        use_tmp: bool = False,
    ) -> None:
        local_host_dir = "/tmp" if use_tmp else "/cache"

        if hasattr(file, "url") and file.url is not None:
            cache_name = hashlib.md5(file.url.geturl().encode()).hexdigest()

            file_name = file.url.geturl().split("/")[-1]
            local_path = f"{local_host_dir}/{cache_name}/{file_name}"
            file_path = Path(local_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            request.urlretrieve(file.url.geturl(), local_path)
        elif file.remote_id is not None or file.path is not None:
            raise NotImplementedError("Remote ID not implemented yet")
            cache_name = (
                file.remote_id
                if file.remote_id
                else hashlib.md5(str(file.path).encode()).hexdigest()
            )
            file_name = file.path.name
            local_path = f"{local_host_dir}/{cache_name}/{file_name}"
            file_path = Path(local_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._progress_download(str(file.path), local_path)
        else:
            raise Exception("File not found, must pass in URL, Path, or Remote ID.")

        if isinstance(file, Directory):
            file.path = Path(f"{local_host_dir}/{cache_name}_dir")

            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(str(file.path))
            return

        file.path = Path(local_path)

    def _create_file_variable(
        self,
        *,
        path_or_url: str | None = None,
        is_directory: bool = False,
        use_tmp: bool = False,
    ) -> File | Directory:
        path: str | None = None
        url: str | None = None

        if validators.url(path_or_url):
            url = str(urllib.parse.urlparse(path_or_url).geturl())
        else:
            path = path_or_url

        if path is None and url is None:
            raise Exception("Must provide either path or url")

        variable: File | Directory | None = None

        if is_directory:
            variable = Directory(path=path, url=url)
        else:
            variable = File(path=path, url=url)

        self._resolve_file_variable_to_local(variable, use_tmp=use_tmp)
        return variable

    def _parse_inputs(
        self, input_data: t.List[run_schemas.RunInput] | None, graph: Graph
    ) -> t.List[t.Any]:
        inputs = []
        if input_data is None:
            input_data = []
            return input_data
        if len(input_data) > 0:
            for item in input_data:
                input_schema = run_schemas.RunInput.parse_obj(item)
                if input_schema.type == run_schemas.RunIOType.file:
                    if input_schema.file_path is None:
                        raise Exception("File path not provided")

                    variable = self._create_file_variable(
                        path_or_url=input_schema.file_path
                    )
                    inputs.append(
                        variable,
                    )

                else:
                    inputs.append(input_schema.value)
        graph_inputs = list(filter(lambda v: v.is_input, graph.variables))

        if len(inputs) != len(graph_inputs):
            raise Exception("Inputs do not match graph inputs")

        final_inputs = []

        for i, (user_input, model_input) in enumerate(zip(inputs, graph_inputs)):
            target_type = model_input.type_class
            if issubclass(target_type, InputSchema) and isinstance(user_input, dict):
                # schema_instance = target_type()
                annotations = target_type.__annotations__
                for key, value in annotations.items():
                    if isinstance(value, UnionType) or "typing.Optional" in str(value):
                        var_union_types = list(t.get_args(value))
                        if len(var_union_types) > 2:
                            raise Exception("Only support Union of 2 types")
                        if NoneType in var_union_types:
                            var_union_types.remove(NoneType)
                        else:
                            raise Exception("Only support Union with None")
                        var_type = var_union_types[0]
                    else:
                        var_type = value

                    if var_type == File:
                        io_schema = run_schemas.RunInput.parse_obj(user_input)
                        if io_schema.value[key] is None:
                            continue

                        file_schema = run_schemas.RunInput.parse_obj(
                            io_schema.value[key]
                        )
                        variable = self._create_file_variable(
                            path_or_url=file_schema.file_path,
                            use_tmp=True,
                        )
                        user_input[key] = variable

            final_inputs.append(user_input)
        return final_inputs

    async def run(self, input_data: t.List[run_schemas.RunInput] | None) -> t.Any:
        args = self._parse_inputs(input_data, self.pipeline)
        return self.pipeline.run(*args)
