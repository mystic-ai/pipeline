from dotenv import load_dotenv

from pipeline import PipelineV2

load_dotenv("hidden.env")

pipeline = PipelineV2("MathsTest")


@pipeline.stage
def multiply(a: float, b: float) -> float:
    return a * b


upload_output = pipeline.upload()
# print(upload_output.functions[0].dict())
