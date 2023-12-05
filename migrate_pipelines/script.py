import base64
import csv
import json
import os

from boto3.session import Session
from botocore.client import Config
from botocore.handlers import set_list_objects_encoding_type_url

from pipeline.console.container import _build_container, _push_container

# Get these from the k8s catalyst secrets of the same name
ACCESS_KEY = base64.b64decode(os.getenv("AWS_ACCESS_KEY_ID")).decode("utf8")
SECRET_KEY = base64.b64decode(os.getenv("AWS_SECRET_ACCESS_KEY")).decode("utf8")

PIPELINE_DATA = """
SELECT pipeline.name, pipeline.path, pipeline.accelerators, pipeline.gpu_memory_min,
environment.python_requirements, count(*) as run_count, pipeline.id
FROM pipeline
JOIN environment ON pipeline.environment_id = environment.id
JOIN run ON run.pipeline_id = pipeline.id
GROUP BY pipeline.name, pipeline.path, pipeline.accelerators, pipeline.gpu_memory_min,
environment.python_requirements, pipeline.id
ORDER BY run_count DESC LIMIT 10
"""

YAML_TEMPLATE = """
runtime:
  container_commands:
    - "apt update -y"
    - "apt install -y git libgl1-mesa-glx ffmpeg gcc"
  python:
    python_version: "3.10"
    python_requirements:
      - "git+https://github.com/mystic-ai/pipeline.git@ph/just-balls-in-holes"
{env_req}
accelerators:
{accelerators}
accelerator_memory: {vram_req}
pipeline_graph: "ported_pipeline:ported_pipeline"
pipeline_name: {pipeline_name}

"""


def parse_pipeline_csv():
    pipelines = []
    with open("./old_csvs/pipelines.csv") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            pl = {k: v for k, v in row.items()}
            pl["accelerators"] = json.loads(pl["accelerators"])
            pipelines.append(pl)
    return pipelines


def parse_environments_csv():
    environments = {}
    with open("./old_csvs/environment.csv") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            en = {k: v for k, v in row.items()}
            environments[en["id"]] = en
    return environments


def parse_meta_csv():
    pipeline_metas = {}
    with open("./old_csvs/pipeline_meta.csv") as csvfile:
        spamreader = csv.reader(csvfile)
        headers = spamreader.__next__()  # skip header
        for row in spamreader:
            pipeline_metas[row[0]] = row
    return pipeline_metas, headers


def parse_pointer_csv():
    pointers = {}
    with open("./old_csvs/pointer.csv") as csvfile:
        spamreader = csv.reader(csvfile)
        headers = spamreader.__next__()  # skip header
        for row in spamreader:
            if row[0] in pointers:
                pointers[row[0]].append(row)
            else:
                pointers[row[0]] = [row]
    return pointers, headers


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


def generate_yaml(name, env_req, accelerators, vram_req):
    yaml_content = YAML_TEMPLATE.format(
        env_req="\n".join(["      - " + env for env in env_req]),
        accelerators="\n".join(["  - " + accelerator for accelerator in accelerators]),
        vram_req=vram_req,
        pipeline_name=name,
    )
    with open("./pipeline.yaml", "w") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    pipelines = parse_pipeline_csv()
    pipeline_metas, headers = parse_meta_csv()
    pointers, pointer_headers = parse_pointer_csv()
    envs = parse_environments_csv()

    with open("./new_csvs/pipeline_meta.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    with open("./new_csvs/pointer.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(pointer_headers)

    for pipeline in pipelines:
        env = envs[pipeline["environment_id"]]
        reqs = env["python_requirements"][1:-1].split(
            ","
        )  # replace {} with [] for json parsing
        print(reqs)

        bucket = get_bucket()
        download_file(bucket, pipeline["path"], "./pipeline.graph")
        generate_yaml(
            pipeline["name"].lower(),
            reqs,
            pipeline["accelerators"],
            pipeline["gpu_memory_min"],
        )
        # _build_container(None)
        # new_id = _push_container(None)
        # print(new_id)

        # meta = pipeline_metas[pipeline["id"]]
        # meta[0] = new_id

        # with open("./new_csvs/pipeline_meta.csv", "a") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(meta)

        # pointers_for_pipeline = pointers[pipeline["id"]]
        # for pt in pointers_for_pipeline:
        #     pt[0] = new_id
        #     pt[2] = pt[2].lower()

        # with open("./new_csvs/pointer.csv", "a") as csvfile:
        #     writer = csv.writer(csvfile)
        #     for pt in pointers_for_pipeline:
        #         writer.writerow(pt)
