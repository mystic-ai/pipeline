import os
import typing as t
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path
from zipfile import ZipFile

import requests
from tabulate import tabulate

from pipeline.cloud import http
from pipeline.cloud.schemas import files as files_schemas
from pipeline.util.logging import _print


def create_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    create_parser = command_parser.add_parser(
        "files",
        aliases=["file"],
        help="Create a new file.",
    )
    create_parser.set_defaults(func=_create_file)

    create_parser.add_argument(
        "path",
        help="Local path to the file.",
    )

    create_parser.add_argument(
        "--name",
        "-n",
        help="Remote alias name of the file.",
        type=str,
    )

    create_parser.add_argument(
        "--recursive",
        "-r",
        help="Recursively upload files in a directory (upload a directory).",
        action="store_true",
    )


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "files",
        aliases=["file"],
        help="Get file information.",
    )
    get_parser.set_defaults(func=_get_file)

    get_parser.add_argument(
        "id",
        nargs="?",
        help="File ID.",
    )

    get_parser.add_argument(
        "--output-file",
        "-o",
        help="Output file path.",
    )

    get_parser.add_argument(
        "--download",
        "-d",
        help="Download file.",
        action="store_true",
    )


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def _create_file(args: Namespace) -> None:
    path: str = getattr(args, "path")
    name: str = getattr(args, "name", None)
    recursive: bool = getattr(args, "recursive", False)

    local_path: Path = Path(path)
    local_path = local_path.expanduser().resolve()

    if not local_path.exists():
        raise FileNotFoundError(f"File {local_path} does not exist.")

    if local_path.is_dir() and not recursive:
        raise FileNotFoundError(
            f"File {local_path} is not a file. (Use --recursive to upload a directory.)"
        )

    if recursive:
        tmp_path = Path("/tmp") / (str(local_path.name) + ".zip")
        with ZipFile(str(tmp_path), "w") as zip_file:
            for root, dirs, files in os.walk(str(local_path)):
                for file in files:
                    zip_file.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), str(local_path)),
                    )
        zip_path = tmp_path

        zip_file = zip_path.open("rb")
        try:
            res = http.post_files(
                "/v3/pipeline_files",
                files=dict(pfile=zip_file),
                progress=True,
            )
        finally:
            zip_file.close()
            print(str(zip_path))
            os.remove(str(zip_path))
        if res is None:
            raise Exception("Failed uploading file")

        if res.status_code != 201:
            raise Exception(
                f"Failed uploading file: {res.text} (code = {res.status_code})"
            )

        res_schema = res.json()
        _print(f"Directory uploaded successfully with ID: {res_schema['id']}")
        return

    query_params = dict()
    if name is not None:
        query_params["name"] = name

    local_file = local_path.open("rb")
    res: requests.Response | None = None
    try:
        res = http.post_files(
            "/v3/pipeline_files",
            files=dict(pfile=local_file),
            progress=True,
            params=query_params,
        )

    finally:
        local_file.close()

    if res is None:
        raise Exception("Failed uploading file")

    if res.status_code != 201:
        raise Exception(f"Failed uploading file: {res.text} (code = {res.status_code})")

    res_schema = res.json()

    _print(f"File uploaded successfully with ID: {res_schema['id']}")


def _get_file(args: Namespace) -> None:
    id: str | None = getattr(args, "id", None)
    output_file: str | None = getattr(args, "output_file", None)
    download: bool = getattr(args, "download", False)

    if download and id is None:
        raise Exception(
            "Cannot download file without ID. (Can only download one file at a time.)"
        )

    if download:
        res = http.get(
            f"/v3/pipeline_files/download/{id}",
        )

        if res.status_code != 200:
            raise Exception(
                f"Failed downloading file: {res.text} (code = {res.status_code})"
            )

        file = res.content
        with open(output_file, "wb") as f:
            f.write(file)

        _print(f"File downloaded successfully to {output_file}")
        return

    if id is not None:
        res = http.get(
            f"/v3/pipeline_files/{id}",
        )

        if res.status_code != 200:
            raise Exception(
                f"Failed getting file: {res.text} (code = {res.status_code})"
            )

        res_schema = res.json()
        res_schema = files_schemas.FileGet.parse_obj(res_schema)

        file = [
            [
                res_schema.id,
                res_schema.created_at,
                res_schema.path,
            ]
        ]

        table = tabulate(
            file,
            headers=[
                "ID",
                "Created At",
                "Path",
            ],
            tablefmt="psql",
        )
        print(table)

        return

    res = http.get(
        "/v3/pipeline_files",
    )

    if res.status_code != 200:
        raise Exception(f"Failed getting files: {res.text} (code = {res.status_code})")

    res_schema: list = res.json()
    res_schema: t.List[files_schemas.FileGet] = [
        files_schemas.FileGet.parse_obj(file_schema) for file_schema in res_schema
    ]

    files = [
        [
            file_schema.id,
            file_schema.created_at,
            file_schema.path,
        ]
        for file_schema in res_schema
    ]

    table = tabulate(
        files,
        headers=[
            "ID",
            "Created At",
            "Path",
        ],
        tablefmt="psql",
    )
    print(table)
