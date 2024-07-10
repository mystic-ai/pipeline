import json
import subprocess
from argparse import Namespace
from pathlib import Path

import yaml

from pipeline.container import docker_templates
from pipeline.util.logging import _print

from .build import build_pipeline_container
from .schemas import Converter, PipelineConfig, PythonRuntime, RuntimeConfig

PIPELINE_INPUT_TEMPLATE = """
    {input_name}: {input_type} | None = InputField(
        title="{title}",
        description="{desc}",
        default={default},
        optional=True,
    )
"""

TYPES_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
}


def convert(namespace: Namespace) -> None:
    framework = namespace.type

    _print(f"Initializing new pipeline from {framework}...", "INFO")

    pipeline_name = getattr(namespace, "name", None)
    if not pipeline_name:
        pipeline_name = input("Enter a name for your pipeline: ")

    if framework == "cog":
        convert_cog(pipeline_name)
    else:
        raise NotImplementedError(f"Framework {framework} not supported")

    _print("Initialized new pipeline.", "SUCCESS")


def convert_cog(pipeline_name: str):

    # first build cog image
    # check cog command exists
    try:
        subprocess.run(["cog", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        _print(
            "cog not found, please install cog first: https://github.com/replicate/cog",
            "ERROR",
        )
        raise

    cog_image_name = f"cog--{pipeline_name}"

    subprocess.run(
        ["cog", "build", "-t", cog_image_name],
        check=True,
        # capture_output=True,
    )

    pipeline_code = _generate_pipeline_code()

    config = PipelineConfig(
        runtime=RuntimeConfig(
            container_commands=[],
            python=PythonRuntime(
                version="3.10",
                requirements=["pipeline-ai", "httpx"],
            ),
        ),
        accelerators=[],
        pipeline_graph="new_pipeline:my_new_pipeline",
        pipeline_name=pipeline_name,
        accelerator_memory=None,
        # TODO - change to using structured field (Converter)
        # converter=Converter(framework="cog", options={"image": cog_image_name}),
        extras={"wrapping": {"framework": "cog", "image": cog_image_name}},
        readme="README.md",
    )

    # write files into a subdirectory so we don't include the cog code inside
    # our pipeline
    subdir = Path("pipeline")
    # raise Exception if dir already exists
    subdir.mkdir(exist_ok=False)
    with open(subdir / "new_pipeline.py", "w") as f:
        f.write(pipeline_code)

    with open(subdir / "pipeline.yaml", "w") as f:
        f.write(yaml.dump(config.dict()))

    with open(subdir / "README.md", "w") as f:
        f.write(docker_templates.readme_template)

    build_pipeline_container(
        config_file_path=None, dockerfile_path=None, base_dir=subdir
    )


def _generate_pipeline_code():
    openapi_schema = Path.cwd() / ".cog" / "openapi_schema.json"
    try:
        with open(openapi_schema) as f:
            schema = json.load(f)
    except FileNotFoundError:
        _print("No .cog/openapi_schema.json found in current directory", "ERROR")
        raise

    inputs = (
        schema.get("components", {})
        .get("schemas", {})
        .get("Input", {})
        .get("properties", {})
    )

    inputs_python = []
    api_inputs = []
    for name, val in inputs.items():
        if "type" not in val:
            _print(f"Skipping input '{name}' since type unknown")
            continue
        python_type = TYPES_MAP.get(val["type"])
        if not python_type:
            raise ValueError(f"Unknown model input type found: {val['type']}")

        default = val.get("default", None)
        if default is not None:
            default = json.dumps(default)
        inputs_python.append(
            PIPELINE_INPUT_TEMPLATE.format(
                input_name=name,
                input_type=python_type,
                desc=val.get("description", ""),
                title=val.get("title", name),
                default=default,
            )
        )
        api_inputs.append(f'"{name}": kwargs.{name}')

    schema_output = schema.get("components", {}).get("schemas", {}).get("Output", {})
    schema_output_type = schema_output.get("type")
    if not schema_output_type:
        raise ValueError("Could not find output type in cog OpenAPI schema")
    if schema_output_type == "array":
        list_schema_type = schema_output.get("items", {}).get("type")
        python_list_type = TYPES_MAP.get(list_schema_type)
        if not python_list_type:
            raise ValueError(f"Unknown model ouput type found: {list_schema_type}")
        python_output_type = f"list[{python_list_type}]"
    else:
        python_output_type = TYPES_MAP.get(schema_output_type)
        if not python_output_type:
            raise ValueError(f"Unknown model ouput type found: {schema_output_type}")

    python_template = docker_templates.pipeline_replicate_template
    pipeline_code = python_template.format(
        input_fields="".join(inputs_python),
        api_inputs=", ".join(api_inputs),
        output_type=python_output_type,
    )
    return pipeline_code
