from typing import List
from pipeline.objects import Paiplain, pipeline


# Check naming
def test_with_decorator_name():
    pipeline = Paiplain("test_instance")
    assert pipeline.pipeline_context_name == "test_instance"


# Test basic Paiplain
def test_basic_pipeline():
    pipeline = Paiplain("basic_test")

    @pipeline.stage
    def add(f_1: float, f_2: float) -> float:
        return f_1 + f_2

    @pipeline.stage
    def square(f_1: float) -> float:
        return f_1 ** 2

    output = pipeline.process(2.0, 3.0)
    results = pipeline.get_results()
    assert results == [5.0, 25.0]
    assert output == pipeline.get_named_results()


def test_set_bulk_stages():
    pipeline = Paiplain("bulk")

    def add(a: float, b: float) -> float:
        return a + b

    def square(a: float) -> float:
        return a ** 2

    def pair(a: float) -> List[float]:
        return [a, a]

    def minus(a: float, b: float) -> float:
        return a - b

    pipeline.set_stages(square, pair, add, pair, minus)
    output = pipeline.process(2.0)
    results = pipeline.get_results()
    print(output)
    assert results == [4.0, [4.0, 4.0], 8.0, [8.0, 8.0], 0.0]
    assert output == pipeline.get_named_results()
