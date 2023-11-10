import base64
import csv
import json
import os

from boto3.session import Session
from botocore.client import Config
from botocore.handlers import set_list_objects_encoding_type_url

# Get these from the k8s catalyst secrets of the same name
ACCESS_KEY = base64.b64decode(os.getenv("AWS_ACCESS_KEY_ID")).decode("utf8")
SECRET_KEY = base64.b64decode(os.getenv("AWS_SECRET_ACCESS_KEY")).decode("utf8")

PIPELINE_DATA = """
SELECT pipeline.name, pipeline.path, pipeline.accelerators, pipeline.gpu_memory_min,
environment.python_requirements, count(*) as run_count
FROM pipeline
JOIN environment ON pipeline.environment_id = environment.id
JOIN run ON run.pipeline_id = pipeline.id
GROUP BY pipeline.name, pipeline.path, pipeline.accelerators, pipeline.gpu_memory_min,
environment.python_requirements
ORDER BY run_count DESC
"""

YAML_TEMPLATE = """
runtime:
  container_commands:
    - "apt update -y"
    - "apt install -y git"
  python:
    python_version: "3.10"
    python_requirements:
      - "git+https://github.com/mystic-ai/pipeline.git@ph/just-balls-in-holes"
{env_req}
accelerators:
{accelerators}
accelerator_memory: {vram_req}
pipeline_graph: "ported_pipeline:ported_pipeline"
pipeline_name: matthew/port

"""


def parse_csv():
    pipelines = []
    with open("./migrate_pipelines/pipelines.csv") as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            pipelines.append(
                {
                    "name": row[0],
                    "path": row[1],
                    "accelerators": json.loads(row[2]),
                    "vram_req": row[3],
                    "env_req": row[4][1:-1].split(","),
                }
            )
    return pipelines


def get_bucket():
    session = Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="europe-west4",
    )
    session.events.unregister(
        "before-parameter-build.s3.ListObjects", set_list_objects_encoding_type_url
    )
    s3 = session.resource(
        "s3",
        endpoint_url="https://storage.googleapis.com",
        config=Config(signature_version="s3v4"),
    )
    bucket = s3.Bucket("pcore-gcp-catalyst-production")
    return bucket


def download_file(bucket, path, dst):
    bucket.download_file(path, dst)


def generate_yaml(env_req, accelerators, vram_req):
    yaml_content = YAML_TEMPLATE.format(
        env_req="\n".join(["      - " + env for env in env_req]),
        accelerators="\n".join(["  - " + accelerator for accelerator in accelerators]),
        vram_req=vram_req,
    )
    with open("./migrate_pipelines/pipeline.yaml", "w") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    pipelines = parse_csv()
    bucket = get_bucket()
    download_file(bucket, pipelines[0]["path"], "./migrate_pipelines/pipeline.graph")
    generate_yaml(
        pipelines[0]["env_req"], pipelines[0]["accelerators"], pipelines[0]["vram_req"]
    )
