from argparse import Namespace
from pathlib import Path

import docker
import docker.errors
import yaml

from pipeline.container import docker_templates
from pipeline.util.logging import _print

from .schemas import PipelineConfig


def build_container(namespace: Namespace):
    _print("Starting build service...", "INFO")
    template = docker_templates.dockerfile_template

    config_file = Path(getattr(namespace, "file", "./pipeline.yaml"))

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
    dockerfile_path = getattr(namespace, "docker_file", None)
    if dockerfile_path is None:
        dockerfile_str = template.format(
            python_version=python_runtime.version,
            python_requirements=(
                " ".join(python_runtime.requirements)
                if python_runtime.requirements
                else ""
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
    else:
        dockerfile_path = Path(dockerfile_path)
    docker_client = docker.APIClient()
    generator = docker_client.build(
        path="./",
        dockerfile=dockerfile_path.absolute(),
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

    new_container.tag(pipeline_repo)
    _print(f"Created tag {pipeline_repo}", "SUCCESS")

    new_container.tag(pipeline_repo, tag=created_image_short_id)
    _print(f"Created tag {pipeline_repo}:{created_image_short_id}", "SUCCESS")
