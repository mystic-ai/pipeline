import json
import os
import subprocess
import sys
import typing as t
from argparse import Namespace
from pathlib import Path

import docker
import yaml
from docker.types import DeviceRequest, LogConfig
from pydantic import BaseModel

from pipeline.cloud import http
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.schemas import cluster as cluster_schemas
from pipeline.cloud.schemas import pipelines as pipelines_schemas
from pipeline.cloud.schemas import registry as registry_schemas
from pipeline.container import docker_templates
from pipeline.util.logging import _print


class PythonRuntime(BaseModel):
    version: str
    requirements: t.List[str] | None

    class Config:
        extra = "forbid"


class RuntimeConfig(BaseModel):
    container_commands: t.List[str] | None
    python: PythonRuntime | None

    class Config:
        extra = "forbid"


class PipelineConfig(BaseModel):
    runtime: RuntimeConfig
    accelerators: t.List[Accelerator] = []
    accelerator_memory: int | None
    pipeline_graph: str
    pipeline_name: str = ""
    description: str | None = None
    readme: str | None = None
    extras: t.Dict[str, t.Any] | None
    cluster: cluster_schemas.PipelineClusterConfig | None = None

    class Config:
        extra = "forbid"


def _up_container(namespace: Namespace):
    _print("Starting container...", "INFO")
    config_file = Path("./pipeline.yaml")

    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")

    config = config_file.read_text()
    pipeline_config_yaml = yaml.load(config, Loader=yaml.FullLoader)

    pipeline_config = PipelineConfig.parse_obj(pipeline_config_yaml)

    pipeline_name = pipeline_config.pipeline_name
    docker_client = docker.from_env()
    lc = LogConfig(
        type=LogConfig.types.JSON,
        config={
            "max-size": "1g",
        },
    )
    volumes: list | None = None

    port = int(getattr(namespace, "port", "14300"))

    run_command = [
        "uvicorn",
        "pipeline.container.startup:create_app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--factory",
    ]

    environment_variables = dict()

    if getattr(namespace, "debug", False):
        run_command.append("--reload")
        current_path = Path("./").expanduser().resolve()
        if volumes is None:
            volumes = []

        volumes.append(f"{current_path}:/app/")
        environment_variables["DEBUG"] = "1"
        environment_variables["LOG_LEVEL"] = "DEBUG"
        environment_variables["FASTAPI_ENV"] = "development"

    if extra_volumes := getattr(namespace, "volume", None):
        if volumes is None:
            volumes = []

        for volume in extra_volumes:
            if ":" not in volume:
                raise ValueError(
                    f"Invalid volume {volume}, must be in format host_path:container_path"  # noqa
                )

            local_path = Path(volume.split(":")[0]).expanduser().resolve()
            container_path = Path(volume.split(":")[1])

            volumes.append(f"{local_path}:{container_path}")

    gpu_ids: list | None = None
    try:
        gpu_ids = [
            f"{i}"
            for i in range(
                0,
                len(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-L",
                        ]
                    )
                    .decode()
                    .splitlines()
                ),
            )
        ]
    except Exception:
        gpu_ids = None

    # Stop container on python exit
    container = docker_client.containers.run(
        pipeline_name,
        ports={f"{port}/tcp": int(port)},
        stderr=True,
        stdout=True,
        log_config=lc,
        remove=True,
        auto_remove=True,
        detach=True,
        device_requests=(
            [DeviceRequest(device_ids=gpu_ids, capabilities=[["gpu"]])]
            if gpu_ids
            else None
        ),
        command=run_command,
        volumes=volumes,
        environment=environment_variables,
    )

    _print(
        f"Container started on port {port}.\n\n\t\tView the live docs:\n\n\t\t\t http://localhost:{port}/redoc\n\n\t\tor live play:\n\n\t\t\t http://localhost:{port}/play\n",  # noqa
        "SUCCESS",
    )

    while True:
        try:
            for line in container.logs(stream=True):
                print(line.decode("utf-8").strip())
        except KeyboardInterrupt:
            _print("Stopping container...", "WARNING")
            container.stop()
            # container.remove()
            break
        except docker.errors.NotFound:
            _print("Container did not start successfully", "ERROR")
            break


def _build_container(namespace: Namespace):
    _print("Starting build service...", "INFO")
    template = docker_templates.dockerfile_template

    config_file = Path("./pipeline.yaml")

    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")

    config = config_file.read_text()
    pipeline_config_yaml = yaml.load(config, Loader=yaml.FullLoader)

    pipeline_config = PipelineConfig.parse_obj(pipeline_config_yaml)

    if not pipeline_config.runtime:
        raise ValueError("No runtime config found")
    if not pipeline_config.runtime.python:
        raise ValueError("No python runtime config found")

    python_runtime = pipeline_config.runtime.python
    dockerfile_str = template.format(
        python_version=python_runtime.version,
        python_requirements=(
            " ".join(python_runtime.requirements) if python_runtime.requirements else ""
        ),
        container_commands="".join(
            [
                "RUN " + command + " \n"
                for command in pipeline_config.runtime.container_commands or []
            ]
        ),
        pipeline_path=pipeline_config.pipeline_graph,
        pipeline_name=pipeline_config.pipeline_name,
        pipeline_image=pipeline_config.pipeline_name,
    )

    dockerfile_path = Path("./pipeline.dockerfile")
    dockerfile_path.write_text(dockerfile_str)
    docker_client = docker.APIClient()
    generator = docker_client.build(
        # fileobj=dockerfile_path.open("rb"),
        path="./",
        # custom_context=True,
        dockerfile=dockerfile_path.absolute(),
        # tag="test",
        rm=True,
        decode=True,
        platform="linux/amd64",
    )
    docker_image_id = None
    while True:
        try:
            output = generator.__next__()
            if "aux" in output:
                docker_image_id = output["aux"]["ID"]
            if "stream" in output:
                _print(output["stream"].strip("\n"))
            if "errorDetail" in output:
                raise Exception(output["errorDetail"])
        except StopIteration:
            _print("Docker image build complete.")
            break

    docker_client = docker.from_env()
    new_container = docker_client.images.get(docker_image_id)

    created_image_full_id = new_container.id.split(":")[1]
    created_image_short_id = created_image_full_id[:12]

    _print(f"Built container {created_image_short_id}", "SUCCESS")

    pipeline_repo = (
        pipeline_config.pipeline_name.split(":")[0]
        if ":" in pipeline_config.pipeline_name
        else pipeline_config.pipeline_name
    )
    # pipeline_tag = (
    #     pipeline_config.pipeline_name.split(":")[1]
    #     if ":" in pipeline_config.pipeline_name
    #     else None
    # )

    new_container.tag(pipeline_repo)
    _print(f"Created tag {pipeline_repo}", "SUCCESS")

    new_container.tag(pipeline_repo, tag=created_image_short_id)
    _print(f"Created tag {pipeline_repo}:{created_image_short_id}", "SUCCESS")

    # if pipeline_tag:
    #     new_container.tag(pipeline_repo, tag=pipeline_tag)
    #     _print(f"Created tag {pipeline_repo}:{pipeline_tag}", "SUCCESS")


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

    # Check for file, transform to string, and put it back in config
    if pipeline_config.readme is not None:
        if os.path.isfile(pipeline_config.readme):
            markdown_file = Path(pipeline_config.readme)
            pipeline_config.readme = markdown_file.read_text()
    else:
        pipeline_config.readme = ""

    # Attempt to format the readme
    readmeless_config = pipeline_config.copy()
    readmeless_config.readme = None
    pipeline_yaml_text = yaml.dump(
        json.loads(readmeless_config.json(exclude_none=True, exclude_unset=True)),
        sort_keys=False,
    )

    pipeline_yaml_text = "```yaml\n" + pipeline_yaml_text + "\n```"
    pipeline_code = Path(
        pipeline_config.pipeline_graph.split(":")[0] + ".py"
    ).read_text()
    pipeline_code = "```python\n" + pipeline_code + "\n```"
    pipeline_config.readme = pipeline_config.readme.format(
        pipeline_name=pipeline_config.pipeline_name,
        pipeline_yaml=pipeline_yaml_text,
        pipeline_code=pipeline_code,
    )

    pipeline_name = (
        pipeline_config.pipeline_name.split(":")[0]
        if ":" in pipeline_config.pipeline_name
        else pipeline_config.pipeline_name
    )

    docker_client = docker.from_env(timeout=300)

    registry_info = http.get(endpoint="/v4/registry")
    registry_info = registry_schemas.RegistryInformation.parse_raw(registry_info.text)

    upload_registry = registry_info.url

    if upload_registry is None:
        raise ValueError("No upload registry found")
    image = docker_client.images.get(pipeline_name)
    image_hash = image.id.split(":")[1]

    hash_tag = image_hash[:12]
    image_to_push = pipeline_name + ":" + hash_tag
    image_to_push_reg = upload_registry + "/" + image_to_push

    upload_token = None
    true_pipeline_name = None
    if registry_info.special_auth:
        start_upload_response = http.post(
            endpoint="/v4/registry/start-upload",
            json_data={
                "pipeline_name": pipeline_name,
                "pipeline_tag": None,
            },
        )
        start_upload_dict = start_upload_response.json()
        upload_token = start_upload_dict.get("bearer", None)
        true_pipeline_name = start_upload_dict.get("pipeline_name")

        if upload_token is None:
            raise ValueError("No upload token found")

        # Login to upload registry
        docker_client.login(
            username="pipeline",
            password=upload_token,
            registry="http://" + upload_registry,
        )

        # Override the tag with the pipeline name from catalyst
        image_to_push = true_pipeline_name + ":" + hash_tag
        image_to_push_reg = upload_registry + "/" + image_to_push

    _print(f"Pushing image to upload registry {upload_registry}", "INFO")

    docker_client.images.get(pipeline_name).tag(image_to_push_reg)
    # Do this after tagging, because we need to use
    # the old pipeline name to tag the local image
    if true_pipeline_name:
        pipeline_name = true_pipeline_name

    resp = docker_client.images.push(
        image_to_push_reg,
        stream=True,
        decode=True,
        auth_config=(
            dict(username="pipeline", password=upload_token) if upload_token else None
        ),
    )

    all_ids = []

    current_index = 0

    for line in resp:
        if "error" in line:
            if line["error"] == "unauthorized: authentication required":
                print(
                    """
Failed to authenticate with the registry.
This can happen if your pipeline took longer than an hour to push.
Please try reduce the size of your pipeline or contact mystic.ai"""
                )
                return
            raise ValueError(line["error"])
        elif "status" in line:
            if "id" not in line or line["status"] != "Pushing":
                continue

            if "id" in line and line["id"] not in all_ids:
                all_ids.append(line["id"])
                print("Adding")

            index_difference = all_ids.index(line["id"]) - current_index

            print_string = (
                line["id"] + " " + line["progress"].replace("\n", "").replace("\r", "")
            )

            if index_difference > 0:
                print_string += "\n" * index_difference + "\r"
                # print("up")
            elif index_difference < 0:
                print_string += "\033[A" * abs(index_difference) + "\r"
                # print("down")
            else:
                print_string += "\r"
                # print("same")
            current_index = all_ids.index(line["id"])

            sys.stdout.write(print_string)
            sys.stdout.flush()

    new_deployment_request = http.post(
        endpoint="/v4/pipelines",
        json_data=json.loads(
            pipelines_schemas.PipelineCreate(
                name=pipeline_name,
                image=image_to_push_reg,
                input_variables=[],
                output_variables=[],
                minimum_cache_number=None,
                maximum_cache_number=None,
                gpu_memory_min=pipeline_config.accelerator_memory,
                accelerators=pipeline_config.accelerators,
                description=pipeline_config.description,
                readme=pipeline_config.readme,
                extras=pipeline_config.extras,
                cluster=pipeline_config.cluster,
            ).json()
        ),
    )

    new_deployment = pipelines_schemas.PipelineGet.parse_obj(
        new_deployment_request.json()
    )

    _print(
        f"Created new pipeline deployment for {new_deployment.name} -> {new_deployment.id} (image={new_deployment.image})",  # noqa
        "SUCCESS",
    )


def _init_dir(namespace: Namespace) -> None:
    _print("Initializing directory...", "INFO")

    pipeline_name = getattr(namespace, "name", None)

    if not pipeline_name:
        pipeline_name = input("Enter a name for your pipeline: ")

    python_template = docker_templates.pipeline_template_python

    default_config = PipelineConfig(
        runtime=RuntimeConfig(
            container_commands=[
                "apt-get update",
                "apt-get install -y git",
            ],
            python=PythonRuntime(
                version="3.10",
                requirements=[
                    "pipeline-ai",
                ],
            ),
        ),
        accelerators=[],
        pipeline_graph="new_pipeline:my_new_pipeline",
        pipeline_name=pipeline_name,
        accelerator_memory=None,
        extras={},
        readme="README.md",
    )
    with open("./pipeline.yaml", "w") as f:
        f.write(yaml.dump(json.loads(default_config.json()), sort_keys=False))

    with open("./new_pipeline.py", "w") as f:
        f.write(python_template)

    with open("./README.md", "w") as f:
        f.write(docker_templates.readme_template)

    _print("Initialized directory.", "SUCCESS")
