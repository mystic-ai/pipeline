import argparse
import re
from pathlib import Path
from typing import List

from pip._internal.commands.freeze import freeze

from pipeline import PipelineCloud
from pipeline.schemas.environment import EnvironmentCreate, EnvironmentGet
from pipeline.util.logging import _print

environment_re_pattern = re.compile(r"^[0-9a-zA-Z]+[0-9a-zA-Z\_\-]+[0-9a-zA-Z]+$")


def _get_environment(name_or_id: str) -> EnvironmentGet:

    remote_service = PipelineCloud()
    remote_service.authenticate()

    environment_information = EnvironmentGet.parse_obj(
        remote_service._get(f"/v2/environments/{name_or_id}")
    )

    return environment_information


def _get_packages_from_requirements(requirements_path: str) -> List[str]:
    requirements_path: Path = Path(requirements_path)
    if not requirements_path.exists():
        raise Exception(f"Requirements file does not exist: {requirements_path}")

    raw_packages = requirements_path.read_text().splitlines()

    return raw_packages


def _get_packages_from_local_env() -> List[str]:
    packages = [dep for dep in freeze.freeze() if dep.split() > 0]
    return packages


def _create_environment(name: str, packages: List[str]) -> EnvironmentGet:
    remote_service = PipelineCloud()
    remote_service.authenticate()
    # TODO: Check for local, editable, or private git repo packages
    create_schema = EnvironmentCreate(name=name, python_requirements=packages)

    environment_dict = remote_service._post(
        "/v2/environments", json_data=create_schema.dict()
    )
    return EnvironmentGet.parse_obj(environment_dict)


def environments(args: argparse.Namespace) -> int:

    sub_command = getattr(args, "sub-command", None)

    if sub_command == "create":
        name = getattr(args, "name", None)
        requirements_path = getattr(args, "requirements", None)
        from_local = getattr(args, "from_local")

        if requirements_path is not None and from_local:
            raise Exception("Cannot build from local packages and a requirements file")

        packages = (
            _get_packages_from_local_env()
            if from_local
            else (
                _get_packages_from_requirements(requirements_path)
                if requirements_path is not None
                else []
            )
        )

        new_environment = _create_environment(name, packages)
        _print(
            f"Created new environment '{new_environment.name}' ({new_environment.id})"
        )

    elif sub_command == "get":
        raise NotImplementedError()
    elif sub_command in ["list", "ls"]:
        raise NotImplementedError()
    elif sub_command in ["delete", "rm"]:
        raise NotImplementedError()
    elif sub_command == "create":
        raise NotImplementedError()
