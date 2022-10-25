import os
import shutil
import subprocess
import venv
from typing import Any, List

import cloudpickle
import tomli

from pipeline import config
from pipeline.objects.graph import Graph
from pipeline.util.logging import _print

"""
TODO:
1.  Add in dependency checks to check if a set of deps can be installed.
    This is a solved problem and we can use the internal testing from pip:

    from pip._internal.req.req_install import InstallRequirement
    from pip._vendor.packaging.requirements import Requirement
    from pip._internal.operations import check
"""


class Dependency:
    def __init__(self, dependency_string) -> None:
        self.dependency_string = dependency_string


class Environment:

    initialized: bool = False

    def __init__(
        self, environment_name: str = None, dependencies: List[Dependency] = None
    ):
        self.environment_name = environment_name
        self.dependencies = dependencies

    def initialize(self, *, overwrite: bool = False, upgrade_deps: bool = True) -> str:
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
        self.env_path = os.path.join(config.PIPELINE_CACHE, self.environment_name)

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
                    f"Deleting existing '{self.environment_name}' env",
                    "WARNING",
                )
                shutil.rmtree(self.env_path)

        # TODO change this to main
        self.add_dependency(Dependency("/Users/paul/mystic/pipeline-stack/pipeline"))
        self.add_dependency(Dependency("dill"))

        venv.create(
            env_dir=self.env_path,
            clear=True,
            with_pip=True,
            upgrade_deps=upgrade_deps,
        )

        # Create requirements.txt for env
        requirements_path = os.path.join(self.env_path, "requirements.txt")
        with open(requirements_path, "w") as req_file:
            for _dep in self.dependencies:
                req_file.write(f"{_dep.dependency_string}\n")

        env_python_path = os.path.join(self.env_path, "bin/python")
        subprocess.call(
            [env_python_path, "-m", "pip", "install", "-r", requirements_path],
            stdout=subprocess.PIPE,
        )
        _print(f"New environment '{self.environment_name}' has been created")
        self.initialized = True

    def add_dependency(self, dependency: Dependency) -> None:
        if self.initialized:
            raise Exception(
                "Cannot add dependency after the environment has \
                been initialized."
            )
        self.dependencies.append(dependency)

    @classmethod
    def from_requirements(cls, requirements_path: str, environment_name: str = None):
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(
                f"Could not find the requirements file '{requirements_path}'"
            )

        with open(requirements_path, "r") as req_file:
            requirements_str_list = req_file.readlines()

        requirements_list = [
            Dependency(_req.trim())
            for _req in requirements_str_list
            if not _req.startswith("#")
        ]
        return cls(environment_name=environment_name, dependencies=requirements_list)

    @classmethod
    def from_toml(
        cls,
        toml_path: str,
        environment_name: str = None,
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
        return cls(environment_name=environment_name, dependencies=requirements_list)


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
            raise Exception("Worker couldn't start")
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
