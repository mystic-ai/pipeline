from httpx import HTTPStatusError

from pipeline.cloud import http
from pipeline.util.logging import _print


def create_environment(
    name: str,
    python_requirements: list[str],
    allow_existing: bool = True,
) -> int:
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
