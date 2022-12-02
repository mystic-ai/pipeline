import copy
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import yaml

from pipeline.objects import Graph, PipelineFile
from pipeline.objects.environment import Environment


def create_pipeline_api(
    pipeline_graphs: List[Graph],
    *,
    output_dir: str = "./",
    platform: str = "linux/amd64",
    environment: Optional[Environment] = None,
    gpu_index: Optional[str] = None,
    **environment_variables,
):
    paths = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline_graphs = copy.deepcopy(pipeline_graphs)

    local_file_paths: List[str] = []

    for pipeline_graph in pipeline_graphs:
        for _var in pipeline_graph.variables:
            if isinstance(_var, PipelineFile):
                local_file_paths.append(_var.path)
                _var.path = os.path.join("/app/files/", os.path.basename(_var.path))

        graph_path = os.path.join(output_dir, pipeline_graph.name + ".graph")
        pipeline_graph.save(graph_path)
        paths.append(graph_path)
    if environment is not None:
        environment.to_requirements()
    create_dockerfile(
        paths,
        output_dir=output_dir,
        platform=platform,
        requirements="requirements.txt",
        pipeline_file_paths=local_file_paths,
    )

    create_docker_compose(output_dir, gpu_index=gpu_index, **environment_variables)


def create_dockerfile(
    pipeline_graph_paths: List[str],
    *,
    output_dir: str = "./",
    platform: str = "linux/amd64",
    requirements: Optional[str] = None,
    pipeline_file_paths: Optional[List[str]] = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, "Dockerfile"), "w") as docker_file:
        docker_file.writelines(
            [
                "FROM --platform=%s mysticai/pipeline-docker:latest" % platform,
            ]
        )

        if requirements is not None:
            docker_file.writelines(
                [
                    f"\nCOPY {requirements} /app/requirements.txt",
                    "\nRUN pip install -r requirements.txt",
                ]
            )
        docker_file.writelines(
            ["\nCOPY %s /app/pipelines/" % path for path in pipeline_graph_paths]
        )

        if pipeline_file_paths is not None:
            docker_file.writelines(
                ["\nCOPY %s /app/files/" % path for path in pipeline_file_paths]
            )


def create_docker_compose(path, gpu_index: Optional[str] = None, **environment_vars):
    presets_env_vars = {
        "LOG_LEVEL": "debug",
        "PYTHONDONTWRITEBYTECODE": 1,
        "PYTHONUNBUFFERED": 1,
        "DATABASE_HOSTNAME": "pipeline_db",
        "DATABASE_PORT": 5432,
        "DATABASE_USERNAME": "postgres",
        "DATABASE_PASSWORD": "example",
        "DATABASE_NAME": "pipeline",
        "DATABASE_CREATE_TABLES": 1,
        "REDIS_URL": "pipeline_redis",
    }

    for key in presets_env_vars:
        if key not in environment_vars:
            environment_vars[key] = presets_env_vars[key]

    docker_yml_dict = {
        "version": "3.9",
        "services": {
            "pipeline_api": {
                "image": "mysticai/pipeline-docker:latest",
                "container_name": "pipeline_api",
                "restart": "always",
                "ports": ["5010:80"],
                "build": {"context": "./", "dockerfile": "Dockerfile"},
                "environment": {
                    **environment_vars,
                },
                "depends_on": ["pipeline_db"],
            },
            "pipeline_db": {
                "image": "postgres",
                "container_name": "pipeline_db",
                "restart": "always",
                "environment": {"POSTGRES_PASSWORD": "example"},
            },
            "pipeline_redis": {
                "image": "bitnami/redis:latest",
                "container_name": "pipeline_redis",
                "restart": "always",
                "environment": ["ALLOW_EMPTY_PASSWORD=yes"],
                "ports": ["6379:6379"],
            },
        },
    }

    if gpu_index is not None:
        docker_yml_dict["services"]["pipeline_api"]["deploy"] = {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "device_ids": [f"{gpu_index}"],
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }
        }

    with open(os.path.join(path, "docker-compose.yml"), "w") as docker_compose_file:
        docker_compose_file.write(yaml.dump(docker_yml_dict))


def build():
    subprocess.run("sudo docker compose pull".split())
    subprocess.run("sudo docker compose build".split())
