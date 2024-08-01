from io import BytesIO
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import docker
import docker.errors
import yaml

from pipeline.cloud import http
import httpx
from pipeline.cloud.schemas import cluster as cluster_schemas
from pipeline.cloud.schemas import pipelines as pipelines_schemas
from pipeline.cloud.schemas import registry as registry_schemas
from pipeline.util.logging import _print

from .pointer import create_pointer
from .schemas import PipelineConfig


def push_container(namespace: Namespace):
    """
    Upload protocol:
    1. Request upload URL from server, along with auth token
    2. Upload to URL with auth token
    3. Send complete request to server
    """

    config_file = Path(getattr(namespace, "file", "./pipeline.yaml"))

    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")

    config = config_file.read_text()
    pipeline_config_yaml = yaml.load(config, Loader=yaml.FullLoader)

    pointers: list | None = getattr(namespace, "pointer", None)

    pipeline_config = PipelineConfig.parse_obj(pipeline_config_yaml)

    # parse optional cluster CLI arg
    cluster_id: str | None = getattr(namespace, "cluster", None)
    node_pool_id: str | None = getattr(namespace, "node_pool", None)
    # Ensure node pool arg is provided if cluster arg provided
    if cluster_id and node_pool_id is None:
        raise ValueError("If --cluster is provided, --node-pool must also be provided")
    if node_pool_id and cluster_id is None:
        raise ValueError("If --node-pool is provided, --cluster must also be provided")
    if cluster_id and node_pool_id:
        pipeline_config.cluster = cluster_schemas.PipelineClusterConfig(
            id=cluster_id, node_pool=node_pool_id
        )

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

    docker_client: docker.DockerClient = docker.from_env(timeout=600)

    registry_params = {}
    turbo_registry = pipeline_config.extras and pipeline_config.extras.get(
        "turbo_registry", False
    )
    if turbo_registry:
        registry_params["turbo_registry"] = "yes"

    registry_info = http.get(endpoint="/v4/registry", params=registry_params)
    registry_info = registry_schemas.RegistryInformation.parse_raw(registry_info.text)

    upload_registry = registry_info.url

    if upload_registry is None:
        raise ValueError("No upload registry found")
    image = docker_client.images.get(pipeline_name)
    image_hash = image.id.split(":")[1]

    if turbo_registry:
        if len(image.attrs["RootFS"]["Layers"]) > 1:
            _print(
                f"Turbo registry image contains multiple layers, this will cause issues with cold start optimization. Please contact Mystic support at support@mystic.ai",  # noqa
                "ERROR",
            )
            raise Exception("Failed to push")

    hash_tag = image_hash[:12]
    image_to_push = pipeline_name + ":" + hash_tag
    image_to_push_reg = upload_registry + "/" + image_to_push

    upload_token = None
    true_pipeline_name = None
    if registry_info.special_auth:
        start_upload_response = http.post(
            endpoint="/v4/registry/start-upload",
            json_data=pipelines_schemas.PipelineStartUpload(
                pipeline_name=pipeline_name,
                pipeline_tag=None,
                cluster=pipeline_config.cluster,
                turbo_registry=True if turbo_registry else False,
                accelerators=pipeline_config.accelerators,
            ).dict(),
        )
        start_upload_dict = start_upload_response.json()
        upload_token = start_upload_dict.get("bearer", None)
        true_pipeline_name = start_upload_dict.get("pipeline_name")

        if upload_token is None:
            raise ValueError("No upload token found")

        # Login to upload registry
        try:
            docker_client.login(
                username="pipeline",
                password=upload_token,
                registry="http://" + upload_registry,
            )
        except Exception as e:
            _print(f"Failed to login to registry: {e}", "ERROR")
            raise

        _print(f"Successfully logged in to registry {upload_registry}")

        # Override the tag with the pipeline name from catalyst
        image_to_push = true_pipeline_name + ":" + hash_tag
        image_to_push_reg = upload_registry + "/" + image_to_push

    _print(f"Pushing image to upload registry {upload_registry}", "INFO")

    docker_client.images.get(pipeline_name).tag(image_to_push_reg)
    # Do this after tagging, because we need to use
    # the old pipeline name to tag the local image
    if true_pipeline_name:
        pipeline_name = true_pipeline_name

    _push_docker_image(
        docker_client=docker_client,
        image=image_to_push_reg,
        upload_token=upload_token,
    )

    new_deployment_request = http.post(
        endpoint="/v4/pipelines",
        json_data=pipelines_schemas.PipelineCreate(
            name=pipeline_name,
            image=image_to_push_reg,
            input_variables=[],
            output_variables=[],
            accelerators=pipeline_config.accelerators,
            description=pipeline_config.description,
            readme=pipeline_config.readme,
            extras=pipeline_config.extras,
            cluster=pipeline_config.cluster,
            scaling_config=pipeline_config.scaling_config_name,
        ).dict(),
    )

    new_deployment = pipelines_schemas.PipelineGet.parse_obj(
        new_deployment_request.json()
    )

    _print(
        f"Created new pipeline deployment for {new_deployment.name} -> {new_deployment.id} (image={new_deployment.image})",  # noqa
        "SUCCESS",
    )

    if pointers:
        pointer_overwrite = getattr(namespace, "pointer_overwrite", False)
        for pointer in pointers:
            create_pointer(pointer, new_deployment.id, force=pointer_overwrite)


def _push_docker_image(
    docker_client: docker.DockerClient,
    image: str,
    upload_token: str,
):
    registry, repo = image.split("/")
    repo = repo.split(":")[0]
    print(f"http://{registry}/v3/{repo}")

    image_obj = docker_client.images.get(image)
    # image_data = b""
    f = BytesIO()
    for dat in image_obj.save(named=True, chunk_size=279396563):
        f.write(dat)
    print("saved")

    f.seek(0)

    httpx.post(
        f"http://{registry}/v3/{repo}",
        data=f.read(),
        headers={"Authorization": "Basic dXNlcjpwYXNzd29yZA=="},
    )
    #     resp = docker_client.images.push(
    #         image,
    #         stream=True,
    #         decode=True,
    #         auth_config=(
    #             dict(username="pipeline", password=upload_token) if upload_token else None
    #         ),
    #     )

    #     all_ids = []

    #     current_index = 0

    #     for line in resp:
    #         if "error" in line:
    #             if line["error"] == "unauthorized: authentication required":
    #                 print(
    #                     """
    # Failed to authenticate with the registry.
    # This can happen if your pipeline took longer than an hour to push.
    # Please try reduce the size of your pipeline or contact mystic.ai"""
    #                 )
    #                 return
    #             raise ValueError(line["error"])
    #         elif "status" in line:
    #             if "id" not in line or line["status"] != "Pushing":
    #                 continue

    #             if "id" in line and line["id"] not in all_ids:
    #                 all_ids.append(line["id"])
    #                 print("Adding")

    #             index_difference = all_ids.index(line["id"]) - current_index

    #             print_string = (
    #                 line["id"] + " " + line["progress"].replace("\n", "").replace("\r", "")
    #             )

    #             if index_difference > 0:
    #                 print_string += "\n" * index_difference + "\r"
    #                 # print("up")
    #             elif index_difference < 0:
    #                 print_string += "\033[A" * abs(index_difference) + "\r"
    #                 # print("down")
    #             else:
    #                 print_string += "\r"
    #                 # print("same")
    #             current_index = all_ids.index(line["id"])

    #             sys.stdout.write(print_string)
    #             sys.stdout.flush()

    _print(f"Successfully pushed image '{image}' to registry")
