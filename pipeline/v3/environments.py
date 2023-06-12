from pipeline.v3 import http


def create_environment(name: str, python_requirements: list[str]) -> int:
    res = http.post(
        "/v3/environments",
        json_data={"name": name, "python_requirements": python_requirements},
    )
    env_id = res.json()["id"]
    return env_id
