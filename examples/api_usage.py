from dotenv import load_dotenv

from pipeline import Paiplain

load_dotenv("hidden.env")

pipeline = Paiplain("MathsTest")


@pipeline.stage
def multiply(a: float, b: float) -> float:
    return a * b


upload_output = pipeline.upload()
# print(upload_output.functions[0].dict())
