from pipeline.api.call import post_file


def upload_file(file_or_path, remote_path):
    if isinstance(file_or_path, str):
        with open(file_or_path, "rb") as file:
            return post_file("/v2/files/", file, remote_path)
    else:
        return post_file("/v2/files/", file_or_path, remote_path)
