from typing import List

from pipeline import PipelineV2


# Test maths Paiplain
def test_maths_pipeline():
    pipeline = PipelineV2("maths")

    @pipeline.stage
    def minus(a: float, b: float) -> float:
        return a - b

    @pipeline.stage
    def square(a: float) -> float:
        return a ** 2

    @pipeline.stage
    def pair(a: float) -> List[float]:
        return [a, a]

    @pipeline.stage
    def multiply(a: float, b: float) -> float:
        return a * b

    output = pipeline.run(4.0, 2.0)
    results = pipeline.get_results()
    assert results == [2.0, 4.0, [4.0, 4.0], 16.0]
    assert output == pipeline.get_named_results()
