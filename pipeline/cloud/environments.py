import time

from httpx import HTTPStatusError

from pipeline.cloud import http
from pipeline.cloud.schemas.environments import EnvironmentGet
from pipeline.cloud.schemas.environments import EnvironmentVerificationStatus as EVS
from pipeline.configuration import current_configuration
from pipeline.util.logging import _print


def create_environment(
    name: str,
    python_requirements: list[str],
    allow_existing: bool = True,
) -> EnvironmentGet:
    try:
        res = http.post(
            "/v3/environments",
            json_data={"name": name, "python_requirements": python_requirements},
        )

        environment = EnvironmentGet.parse_obj(res.json())
        _print(
            f"Created environment '{name}' with ID = {environment.id}", level="SUCCESS"
        )

    except HTTPStatusError as e:
        if e.response.status_code == 409 and allow_existing:
            try:
                res = http.get(
                    f"/v3/environments/by-any/{name}",
                )

                environment = EnvironmentGet.parse_obj(res.json())

                _print(
                    f"Using existing environment {name} with ID = {environment.id}",
                    level="INFO",
                )
            except HTTPStatusError as e2:
                if e2.response.status_code == 404:
                    raise Exception(f"Environment {name} does not exist")
                elif e2.response.status_code == 401:
                    # raise Exception(f"Environment {name} is not accessible to you.")
                    raise Exception(
                        "Environment already exists and is not accessible to you.",
                    )

                else:
                    raise Exception(
                        f"Unknown error: {e2.response.status_code}, {e2.response.text}"
                    )
        else:
            raise e

    if current_configuration.is_debugging():
        if environment.verification_status == EVS.unverified:
            _print("Verifying environment...", level="INFO")
            environment = verify_environment(environment)
            if environment is None:
                raise Exception("Environment verification failed")

        elif environment.verification_status == EVS.failed:
            _print("Environment verification failed", level="ERROR")
            _print(f"Traceback:{environment.verification_exception}", level="ERROR")
            raise Exception("Environment verification failed")
        elif environment.verification_status == EVS.verified:
            _print("Environment is already verified", level="SUCCESS")
            return environment
        elif environment.verification_status == EVS.verifying:
            _print(
                "Environment is already verifying, pending completion...",
                level="INFO",
            )
            while environment.verification_status == EVS.verifying:
                time.sleep(1)
                environment = get_environment(environment.id)

    _print("Environment verified successfully!", level="SUCCESS")
    return environment


def get_environment(name_or_id: str) -> EnvironmentGet:
    try:
        res = http.get(f"/v3/environments/by-any/{name_or_id}")
        environment = EnvironmentGet.parse_obj(res.json())
    except HTTPStatusError as e:
        if e.response.status_code == 404:
            raise Exception(f"Environment {name_or_id} does not exist")
        elif e.response.status_code == 401:
            raise Exception(f"Environment {name_or_id} is not accessible to you.")
        else:
            raise Exception(
                f"Unknown error: {e.response.status_code}, {e.response.text}"
            )

    return environment


def verify_environment(environment: EnvironmentGet) -> EnvironmentGet | None:
    try:
        http.get(f"/v3/environments/verify/{environment.id}")
    except HTTPStatusError as e:
        if e.response.status_code == 404:
            raise Exception(f"Environment {environment.id} does not exist")
        elif e.response.status_code == 409:
            raise Exception(f"Environment {environment.id} is already being verified")
        elif e.response.status_code == 424:
            failed_json = e.response.json()
            _print(
                f"Environment {environment.id} verification failed:\n"
                f"{failed_json['detail'].get('exception')}",
                level="ERROR",
            )
            return None
        else:
            raise Exception(
                f"Unknown error: {e.response.status_code}, {e.response.text}"
            )

    return get_environment(environment.id)
