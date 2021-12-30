from dotenv import load_dotenv

from pipeline import Paiplain

load_dotenv("../hidden.env")  # must contain TOKEN envvar

pipeline = Paiplain("Addlol")

pipeline.auth()


@pipeline.stage
def add_lol(a: str) -> str:
    return a + " lol"


upload_output = pipeline.upload()
remote_run_output = pipeline.run_remote("Hey")
print(remote_run_output)
