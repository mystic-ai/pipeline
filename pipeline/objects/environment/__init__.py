import os
from typing import List, Union

from pip._internal.commands.freeze import freeze


"""
TODO:
1.  Add in dependency checks to check if a set of deps can be installed.
    This is a solved problem and we can use the internal testing from pip:

    from pip._internal.req.req_install import InstallRequirement
    from pip._vendor.packaging.requirements import Requirement
    from pip._internal.operations import check
"""


class Environment:
    def __init__(
        self,
        environment_name: str = None,
        dependencies: List[str] = [],
        extra_index_urls: List[str] = [],
        extend_environments: List = [],
    ):
        self.environment_name = environment_name
        self.dependencies = dependencies
        self.extra_index_urls = extra_index_urls
        for _env in extend_environments:
            self.merge_with_environment(_env)

    def to_requirements(self, output_dir="./"):
        requirements_path = os.path.join(os.path.join(output_dir, "requirements.txt"))
        with open(requirements_path, "w") as req_file:
            for _dep in self.dependencies:
                req_file.write(f"{_dep}\n")

    def add_dependency(self, dependency: str) -> None:
        if self.initialized:
            raise Exception(
                "Cannot add dependency after the environment has \
                been initialized."
            )
        self.dependencies.append(dependency)

    def add_dependencies(self, *dependencies: List[Union[List[str], str]]):
        for _target in dependencies:
            if isinstance(_target, list) and len(dependencies) == 1:
                ...
            elif isinstance(_target, str):
                ...
            else:
                raise Exception(
                    "Can either add a list of dependencies or an array of dependencies"
                )

    def merge_with_environment(self, env) -> None:
        if not isinstance(env, Environment):
            raise Exception("Can only merge with another environment")

        self.dependencies.extend(env.dependencies)
        self.extra_index_urls.extend(env.extra_index_urls)

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
