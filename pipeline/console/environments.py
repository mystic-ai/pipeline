import argparse
import re
from pathlib import Path
from typing import List

from pip._internal.commands.freeze import freeze
from tabulate import tabulate

from pipeline import PipelineCloud
from pipeline.schemas.environment import (
    EnvironmentCreate,
    EnvironmentGet,
    EnvironmentPatch,
)
from pipeline.schemas.pagination import Paginated
from pipeline.util.logging import _print

environment_re_pattern = re.compile(r"^[0-9a-zA-Z]+[0-9a-zA-Z\_\-]+[0-9a-zA-Z]+$")


def _get_environment(name_or_id: str) -> EnvironmentGet:

    remote_service = PipelineCloud()
    remote_service.authenticate()

    environment_information = EnvironmentGet.parse_obj(
        remote_service._get(f"/v2/environments/{name_or_id}")
    )

    return environment_information


def _list_environments(
    skip: int,
    limit: int,
) -> Paginated[EnvironmentGet]:
    # TODO: Add in more filter fields

    remote_service = PipelineCloud()
    remote_service.authenticate()

    response = remote_service._get(
        "/v2/environments",
        params=dict(
            skip=skip,
            limit=limit,
            order_by="created_at:desc",
        ),
    )

    paginated_environments = Paginated[EnvironmentGet].parse_obj(response)

    return paginated_environments


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
    # TODO: Validate name
    create_schema = EnvironmentCreate(name=name, python_requirements=packages)

    environment_dict = remote_service._post(
        "/v2/environments", json_data=create_schema.dict()
    )
    return EnvironmentGet.parse_obj(environment_dict)


def _delete_environment(name_or_id: str) -> None:
    remote_service = PipelineCloud()
    remote_service.authenticate()

    environment_information = _get_environment(name_or_id)
    remote_service._delete(f"/v2/environments/{environment_information.id}")


def _update_environment(
    name_or_id: str, *, locked: bool = None, python_requirements: List[str] = None
) -> EnvironmentGet:
    if locked is None and python_requirements is None:
        raise Exception(
            "Must un/lock or define python_requirements when updating an Environment"
        )
    remote_service = PipelineCloud()
    remote_service.authenticate()

    environment_information = _get_environment(name_or_id)

    update_schema = EnvironmentPatch(
        python_requirements=python_requirements, locked=locked
    )

    response = remote_service._patch(
        f"/v2/environments/{environment_information.id}",
        update_schema.dict(),
    )
    return EnvironmentGet.parse_obj(response)


def _add_packages_to_environment(
    name_or_id: str, new_packages: List[str]
) -> EnvironmentGet:
    environment_information = _get_environment(name_or_id)

    current_packages = environment_information.python_requirements
    for new_package in new_packages:
        if new_package in current_packages:
            raise Exception(f"Package '{new_package}' is already in environment")

    current_packages.extend(new_packages)

    return _update_environment(python_requirements=current_packages)


def _remove_packages_from_environment(
    name_or_id: str, packages: List[str]
) -> EnvironmentGet:
    environment_information = _get_environment(name_or_id)

    current_packages = environment_information.python_requirements
    for package in packages:
        if package not in current_packages:
            raise Exception(f"Package '{package}' is not in environment")

    [current_packages.remove(package) for package in packages]

    return _update_environment(python_requirements=current_packages)


def _update_environment_lock(name_or_id: str, locked: bool) -> EnvironmentGet:
    environment_information = _get_environment(name_or_id)

    if environment_information.locked == locked:
        raise Exception(f"The environment already has locked={locked}")

    return _update_environment(name_or_id, locked=locked)


def _tabulate(environments: List[EnvironmentGet]) -> str:
    return tabulate(
        [
            [
                enivonment.id,
                enivonment.name,
            ]
            for enivonment in environments
        ],
        headers=["ID", "Name"],
        tablefmt="outline",
    )


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
        return 0

    elif sub_command == "get":
        name_or_id = getattr(args, "name_or_id", None)
        environment = _get_environment(name_or_id)
        print(environment.json())
        return 0
    elif sub_command in ["list", "ls"]:
        paginated_results = _list_environments(
            getattr(args, "skip"),
            getattr(args, "limit"),
        )
        table_string = _tabulate(paginated_results.data)
        print(table_string)
        return 0
    elif sub_command in ["delete", "rm"]:
        name_or_id: str = getattr(args, "name_or_id")
        _delete_environment(name_or_id)
        _print(f"Deleted {name_or_id}")
        return 0
    elif sub_command == "update":
        name_or_id: str = getattr(args, "name_or_id")

        update_command = getattr(args, "environments-update-sub-command")

        if update_command == "add":
            packages = getattr(args, "packages")
            _add_packages_to_environment(name_or_id, packages)
            _print(f"Added packages to {name_or_id}")
            return 0
        elif update_command == "remove":
            packages = getattr(args, "packages")
            _remove_packages_from_environment(name_or_id, packages)
            _print(f"Removed packages from {name_or_id}")
            return 0
        elif update_command == "lock":
            _update_environment_lock(name_or_id, True)
            _print(f"Locked {name_or_id}")
            return 0
        elif update_command == "unlock":
            _update_environment_lock(name_or_id, False)
            _print(f"Unlocked {name_or_id}")
            return 0
