import os
from typing import List

from pip._internal.commands.freeze import freeze


class Environment:
    def __init__(
        self,
        environment_name: str = None,
        dependencies: List[str] = [],
    ):
        self.environment_name = environment_name
        self.dependencies = dependencies

    def to_requirements(self, output_dir: str = "./") -> None:
        requirements_path = os.path.join(os.path.join(output_dir, "requirements.txt"))
        with open(requirements_path, "w") as req_file:
            for _dep in self.dependencies:
                req_file.write(f"{_dep}\n")

    def add_dependency(self, dependency: str) -> None:
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
            _req.trim() for _req in requirements_str_list if not _req.startswith("#")
        ]
        return cls(environment_name=environment_name, dependencies=requirements_list)

    @classmethod
    def from_current(cls, environment_name: str = None):
        deps = [dep for dep in freeze.freeze() if dep.split() > 0]
        return cls(environment_name=environment_name, dependencies=deps)
