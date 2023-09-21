import pkgutil
import tempfile

import pkg_resources
from httpx import HTTPStatusError
from pip_requirements_parser import RequirementsFile

from pipeline.cloud import http
from pipeline.util.logging import _print


def _validate_requirements(requirements: list[str]) -> None:
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write("\n".join(requirements))
        f.seek(0)
        req_file = RequirementsFile.from_file(f.name)

    for req in req_file.requirements:
        if req.line is not None and req.line.startswith("git+"):
            continue

        if not req.is_pinned:
            pkg_name = getattr(req, "name", "")

            try:
                version = pkg_resources.get_distribution(pkg_name).version
            except Exception:
                version = None

            package_version_checks = [
                "==",
                ">=",
                "<=",
                ">",
                "<",
                "!=",
                "~=",
                "~",
                "===",
                "!==",
            ]

            if req.line is not None and any(
                [check in req.line for check in package_version_checks]
            ):
                continue

            raise ValueError(
                f"Requirement {req.name} is not pinned. Please pin all requirements."
                + (
                    ""
                    if version is None
                    else f" Found version {req.name}=={version} locally."
                )
            )
        if req.is_editable:
            raise ValueError(
                f"Requirement {req.name} is editable. Please remove all editable requirements."  # noqa
            )

        if req.is_local_path:
            raise ValueError(
                f"Requirement {req.name} is a local path. Please remove all local path requirements."  # noqa
            )

        # Check for if the package exists locally
        if req.name is not None and not pkgutil.find_loader(req.name):
            _print(
                f"Package {req.name} not found locally, it is highly recommended to install it as this will likely cause serialisation issues when uploading Pipelines.",  # noqa
                level="WARNING",
            )


def create_environment(
    name: str,
    python_requirements: list[str],
    allow_existing: bool = True,
) -> int:
    _validate_requirements(python_requirements)

    try:
        res = http.post(
            "/v3/environments",
            json_data={"name": name, "python_requirements": python_requirements},
        )

        env_id = res.json()["id"]
        _print(f"Created environment '{name}' with ID = {env_id}", level="SUCCESS")
    except HTTPStatusError as e:
        if e.response.status_code == 409 and allow_existing:
            try:
                res = http.get(
                    f"/v3/environments/{name}",
                )
                env_id = res.json()["id"]
                _print(
                    f"Using existing environment {name} with ID = {env_id}",
                    level="INFO",
                )
            except HTTPStatusError as e2:
                if e2.response.status_code == 404:
                    raise Exception(f"Environment {name} does not exist")
                elif e2.response.status_code == 401:
                    # raise Exception(f"Environment {name} is not accessible to you.")
                    _print(
                        "Environment already exists and is not accessible to you.",
                        level="WARNING",
                    )
                    env_id = name
                else:
                    raise Exception(
                        f"Unknown error: {e2.response.status_code}, {e2.response.text}"
                    )
        else:
            raise e

    return env_id
