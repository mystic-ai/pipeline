import typing as t
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

import docker
import yaml
from docker.errors import BuildError
from pydantic import BaseModel

from pipeline.cloud import http
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.configuration import current_configuration
from pipeline.container import docker_templates
from pipeline.util.logging import _print


class PythonRuntime(BaseModel):
    python_version: str
    python_requirements: t.List[str] | None
    cuda_version: str | None = "11.4"

    class Config:
        extra = "forbid"


class RuntimeConfig(BaseModel):
    container_commands: t.List[str] | None
    python: PythonRuntime | None

    class Config:
        extra = "forbid"


class PipelineConfig(BaseModel):
    runtime: RuntimeConfig
    accelerators: t.List[Accelerator]
    pipeline_graph: str
    pipeline_name: str = ""

    class Config:
        extra = "forbid"


def _build_container(namespace: Namespace):
    _print("Starting build service...", "INFO")
    template = docker_templates.template_1

    config_file = Path("./pipeline.yaml")

    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")

    config = config_file.read_text()
    pipeline_config_yaml = yaml.load(config, Loader=yaml.FullLoader)

    pipeline_config = PipelineConfig.parse_obj(pipeline_config_yaml)

    extra_paths = "COPY ./examples/docker/ ./\n"

    if not pipeline_config.runtime:
        raise ValueError("No runtime config found")
    if not pipeline_config.runtime.python:
        raise ValueError("No python runtime config found")

    python_runtime = pipeline_config.runtime.python
    dockerfile_str = template.format(
        python_version=python_runtime.python_version,
        python_requirements=" ".join(python_runtime.python_requirements)
        if python_runtime.python_requirements
        else "",
        container_commands="".join(
            [
                "RUN " + command + " \n"
                for command in pipeline_config.runtime.container_commands or []
            ]
        ),
        pipeline_path=pipeline_config.pipeline_graph,
        extra_paths=extra_paths,
    )

    dockerfile_path = Path("./pipeline.dockerfile")
    dockerfile_path.write_text(dockerfile_str)
    docker_client = docker.from_env()
    try:
        new_container, build_logs = docker_client.images.build(
            # fileobj=dockerfile_path.open("rb"),
            path="../../",
            quiet=True,
            # custom_context=True,
            dockerfile="./examples/docker/pipeline.dockerfile",
            # tag="test",
            rm=True,
        )
    except BuildError as e:
        for info in e.build_log:
            if "stream" in info:
                for line in info["stream"].splitlines():
                    print(line)
            elif "errorDetail" in info:
                print(info["errorDetail"]["message"])
            else:
                print(info)
        raise e

    created_image_full_id = new_container.id.split(":")[1]
    created_image_short_id = created_image_full_id[:12]

    _print(f"Built container {created_image_short_id}", "SUCCESS")

    pipeline_repo = (
        pipeline_config.pipeline_name.split(":")[0]
        if ":" in pipeline_config.pipeline_name
        else pipeline_config.pipeline_name
    )
    pipeline_tag = (
        pipeline_config.pipeline_name.split(":")[1]
        if ":" in pipeline_config.pipeline_name
        else None
    )

    new_container.tag(pipeline_repo)
    _print(f"Created tag {pipeline_repo}", "SUCCESS")

    new_container.tag(pipeline_repo, tag=created_image_short_id)
    _print(f"Created tag {pipeline_repo}:{created_image_short_id}", "SUCCESS")

    if pipeline_tag:
        new_container.tag(pipeline_repo, tag=pipeline_tag)
        _print(f"Created tag {pipeline_repo}:{pipeline_tag}", "SUCCESS")


def _push_container(namespace: Namespace):
    """

    Upload protocol:
    1. Request upload URL from server, along with auth token
    2. Upload to URL with auth token
    3. Send complete request to server

    """

    config_file = Path("./pipeline.yaml")

    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")

    config = config_file.read_text()
    pipeline_config_yaml = yaml.load(config, Loader=yaml.FullLoader)

    pipeline_config = PipelineConfig.parse_obj(pipeline_config_yaml)

    pipeline_name = (
        pipeline_config.pipeline_name.split(":")[0]
        if ":" in pipeline_config.pipeline_name
        else pipeline_config.pipeline_name
    )

    docker_client = docker.from_env()

    # docker_client.images.push(pipeline_name)
    start_upload_response = http.post(
        endpoint="/v4/pipelines/start-upload",
        json_data={
            "pipeline_name": pipeline_name,
            "pipeline_tag": None,
        },
    )
    start_upload_dict = start_upload_response.json()

    upload_registry = start_upload_dict.get("upload_registry", None)
    upload_token = start_upload_dict.get("bearer", None)

    if upload_token is None:
        raise ValueError("No upload token found")

    if upload_registry is None:
        _print("No upload registry found, not doing anything...", "WARNING")
    else:
        # auth_dict = {
        #     "auths": {
        #         upload_registry: {"username": "pipeline", "password": upload_token}
        #     }
        # }
        # auth_dict = dict(username="lol", password=upload_token)
        # auth_dict = {
        #     upload_registry: {
        #         "username": "lol",
        #         "password": upload_token,
        #     }
        # }

        _print(f"Pushing image to upload registry {upload_registry}", "INFO")
        docker_client.images.get(pipeline_name).tag(
            upload_registry + "/" + pipeline_name
        )

        # Login to upload registry
        docker_client.login(
            username="pipeline",
            password=upload_token,
            registry=upload_registry,
        )

        resp = docker_client.images.push(
            upload_registry + "/" + pipeline_name,
            auth_config=upload_token,
            stream=True,
            decode=True,
        )
        for line in resp:
            print(line)
