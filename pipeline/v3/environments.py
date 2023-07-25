from httpx import HTTPStatusError

from pipeline.v3 import http


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
    except HTTPStatusError as e:
        if e.response.status_code == 409 and allow_existing:
            res = http.get(
                f"/v3/environments/{name}",
            )
            env_id = res.json()["id"]
        else:
            raise e
    return env_id
