from ntpath import join
import os
import yaml

from typing import List

from pipeline.objects import Graph


def create_pipeline_api(
    pipeline_graphs: List[Graph], *, output_dir="./", **environment_variables
):
    paths = []

    for pipeline_graph in pipeline_graphs:
        graph_path = pipeline_graph.name + ".graph"
        pipeline_graph.save(graph_path)
        paths.append(graph_path)

    create_dockerfile(paths, output_dir=output_dir)
    create_docker_compose(output_dir, **environment_variables)


def create_dockerfile(pipeline_graph_paths: List[str], *, output_dir="./"):
    with open(os.path.join(output_dir, "Dockerfile"), "w") as docker_file:
        docker_file.writelines(
            ["FROM --platform=linux/amd64 mysticai/pipeline-docker:latest"]
        )
        docker_file.writelines(
            ["\nCOPY %s /app/pipelines/" % path for path in pipeline_graph_paths]
        )


def create_docker_compose(path, **environment_vars):
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
        "version": "3.3",
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
    with open(join(path, "docker-compose.yml"), "w") as docker_compose_file:
        docker_compose_file.write(yaml.dump(docker_yml_dict))
