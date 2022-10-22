import os

import tomli

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
    def __init__(
        self, environment_name: str = None, dependencies: list[Dependency] = None
    ):
        self.environment_name = environment_name
        self.dependencies = dependencies

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
