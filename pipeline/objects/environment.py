import hashlib
import os
import shutil
import subprocess
import venv
from typing import Any, List

import cloudpickle
import tomli
from pip._internal.commands.freeze import freeze

#
# from pip._internal.operations.check import check_install_conflicts
from pip._internal.req.constructors import install_req_from_line

from pipeline import configuration
from pipeline.objects.graph import Graph
from pipeline.util.logging import _print
from pipeline.exceptions.environment import EnvironmentInitializationError


class Environment:

    initialized: bool = False

    def __init__(
        self,
        name: str = None,
        dependencies: List[str] = None,
        extra_index_urls: List[str] = None,
        extend_environments: List = None,
    ):
        self.name = name
        self.dependencies = dependencies or []
        self.extra_index_urls = extra_index_urls or []
        extend_environments = extend_environments or []
        for _env in extend_environments:
            self.merge_with_environment(_env)

    @property
    def env_path(self):
        return configuration.PIPELINE_CACHE / self.name

    @property
    def python_path(self):
        return self.env_path / "bin" / "python"

    @property
    def hash(self) -> str:
        """Generate unique hash for this environment so we can determine when
        two environments are the same.
        """
        # Combine all the info that makes this environment unique
        env_str = "::".join(
            [
                self.name,
                ";".join(self.dependencies),
                ";".join(self.extra_index_urls),
            ]
        )
        return hashlib.sha256(env_str.encode()).hexdigest()

    def validate_requirements(self):
        install_options = []
        for extra_url in self.extra_index_urls:
            install_options.append("--extra-index-url")
            install_options.append(extra_url)
        try:
            # Try parsing each requirement, which should raise an exception if
            # invalid.
            # This just checks the format of each requirement and not whether it
            # can be installed successfully or not.
            [
                install_req_from_line(
                    dep,
                    options=dict(install_options=install_options),
                    user_supplied=True,
                    isolated=True,
                )
                for dep in self.dependencies
            ]
            # This doesn't seem to work reliably
            # _, result = check_install_conflicts(requirements)
        except Exception as exc:
            error_msg = f"Invalid requirements: {exc}"
            _print(error_msg, "ERROR")
            raise EnvironmentInitializationError(error_msg)

    def initialize(self, *, overwrite: bool = False, upgrade_deps: bool = True) -> None:
        # TODO add arg for remaking on dependency change

        """_summary_

        Args:
            overwrite (bool, optional): If set to true then then if a venv exists
            with the same name, it will be erased and replaced with this new one.
            Defaults to False.

            upgrade_deps (bool, optional): If true then the base venv variables
            will be upgraded to the latest on pypi. This will not effect the
            defined env packages set in self.dependencies.
            Defaults to True

        Returns:
            None: Nothing is returned.
        """

        if os.path.exists(self.env_path):
            if not overwrite:
                self.initialized = True
                _print(
                    "Using existing environment, any new dependencies won't"
                    " be installed. Use 'overwrite=True' to overwrite.",
                    "WARNING",
                )
                return
            else:
                _print(
                    f"Deleting existing '{self.name}' env",
                    "WARNING",
                )
                shutil.rmtree(self.env_path)

        _print(f"Creating new virtualenv at {self.env_path}")
        venv.create(
            env_dir=self.env_path,
            clear=True,
            with_pip=True,
            upgrade_deps=upgrade_deps,
        )

        self.validate_requirements()

        # Create requirements.txt for env
        deps_str = "\n".join(self.dependencies)
        _print(f"Installing the following requirements:\n{deps_str}\n\n")
        requirements_path = self.env_path / "requirements.txt"
        with open(requirements_path, "w") as req_file:
            for dep in self.dependencies:
                req_file.write(f"{dep}\n")

        extra_args = []
        for extra_url in self.extra_index_urls:
            extra_args.append("--extra-index-url")
            extra_args.append(extra_url)

        try:
            subprocess.run(
                [
                    str(self.python_path),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(requirements_path),
                    *extra_args,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            _print(f"Error installing requirements: {exc}", "ERROR")
            _print(exc.stderr, "ERROR")
            raise EnvironmentInitializationError(
                f"Error installing requirements: {exc.stderr}"
            )

        _print(f"New environment '{self.name}' has been created")
        self.initialized = True

    def add_dependency(self, dependency: str) -> None:
        self.add_dependencies([dependency])

    def add_dependencies(self, dependencies: List[str]):
        if self.initialized:
            raise Exception(
                "Cannot add dependencies after the environment has been initialized."
            )
        self.dependencies.extend(dependencies)

    def merge_with_environment(self, env) -> None:
        if not isinstance(env, Environment):
            raise Exception("Can only merge with another environment")

        self.dependencies.extend(env.dependencies)
        self.extra_index_urls.extend(env.extra_index_urls)

    @classmethod
    def from_requirements(cls, requirements_path: str, name: str = None):
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(
                f"Could not find the requirements file '{requirements_path}'"
            )

        with open(requirements_path, "r") as req_file:
            requirements_str_list = req_file.readlines()

        requirements_list = [
            _req.trim() for _req in requirements_str_list if not _req.startswith("#")
        ]
        return cls(name=name, dependencies=requirements_list)

    @classmethod
    def from_toml(
        cls,
        toml_path: str,
        name: str = None,
        *,
        dependency_section: str = "tool.poetry.dependencies",
    ):
        if not os.path.exists(toml_path):
            raise FileNotFoundError(f"Could not find the toml file '{toml_path}'")

        with open(toml_path, "rb") as toml_file:
            toml_dict = tomli.load(toml_file)

        # TODO: complete dependency extraction from the TOML.
        # Need to do a bit more research on how people lay this out as we're
        # biased on poetry right now.

        if dependency_section not in toml_dict:
            raise Exception(
                f"The toml file does not contain the expected dependency \
                section '{dependency_section}'. Either change the dependency \
                section variable: 'Environtment.from_toml(..., \
                dependency_section=\"...\")', or add in the correct section."
            )

        requirements_list = []
        return cls(name=name, dependencies=requirements_list)

    @classmethod
    def from_current(cls, name: str = None):
        deps = [dep for dep in freeze.freeze() if dep.split() > 0]
        return cls(name=name, dependencies=deps)


class EnvironmentSession:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def __enter__(self):
        if not self.environment.initialized:
            raise Exception(
                "Must initialise environment before using it. \
                Run 'your_environment_variable.initialize()'"
            )

        env_python_path = os.path.join(self.environment.env_path, "bin/python")

        self._proc = subprocess.Popen(
            [env_python_path, "-m", "pipeline", "worker"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )

        output = self._proc.stdout.readline().decode().strip()
        if output != "worker-started":
            error = self._proc.stderr.read1().decode().strip()
            raise Exception(f"Worker couldn't start: {error}")
        _print("Session started")
        return self

    def __exit__(self, type, value, traceback):
        self._proc.kill()

    def _send_command(self, command: str, data: str) -> str:
        self._proc.stdin.write(f"{command}\n".encode())
        self._proc.stdin.write(f"{data}\n".encode())
        self._proc.stdin.flush()
        output = None
        while not output:
            output = self._proc.stdout.readline().decode().strip()
        return output

    def add_pipeline(self, pipeline: Graph) -> None:
        pickled_pipeline = cloudpickle.dumps(pipeline)
        response = self._send_command("add-pipeline", pickled_pipeline.hex())

        if response != "done":
            raise Exception(f"Couldn't add pipeline, error:'{response}'")

    def run_pipeline(self, pipeline: Graph, data: list) -> Any:
        pickled_run = cloudpickle.dumps(dict(pipeline_id=pipeline.local_id, data=data))
        response = self._send_command("run-pipeline", pickled_run.hex())
        return response


default_worker_environment = Environment(
    name="default-worker-environment",
    dependencies=[
        "torch==1.13.1",
        "torchvision==0.14.1",
        "torchaudio==0.13.1",
        "transformers==4.21.2",
        "opencv-python==4.5.3.56",
        "tensorflow==2.9.1",
        "tensorflow-hub==0.12.0",
        # "detectron2==0.6",
        "deepspeed==0.5.10",
        "seaborn==0.11.2",
        "numpy==1.21.0",
        "Pillow==9.2.0",
        "spacy[cuda113]==3.4.3",
        "onnxruntime-gpu==1.12.1",
        "sentence-transformers==2.2.2",
        "accelerate==0.10.0",
        "diffusers @ git+https://github.com/huggingface/diffusers.git@5755d16868ec3da7d5eb4f42db77b01fac842ea8",
        "xgboost==1.6.2",
        "einops==0.4.1",
        "wandb==0.13.4",
        "scikit-learn==1.1.2",
        "catboost==1.1",
        "pywhisper==1.0.6",
    ],
)


worker_torch_environment = Environment(
    name="worker-torch-environment",
    dependencies=[
        "torch==1.13.0",
        "torchvision==0.14.0",
        "torchaudio==0.13.0",
    ],
)
