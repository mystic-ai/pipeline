import json

import pytest

from pipeline.schemas.compute_requirements import ComputeRequirements
from pipeline.schemas.pipeline import PipelineCreate


@pytest.mark.parametrize(
    "test",
    [
        {
            "comment": "Valid example: neither compute type nor compute requirements",
            "input": {"compute_type": "gpu"},
            "expected_exception": None,
            "expected": {"compute_type": "gpu", "compute_requirements": None},
        },
        {
            "comment": "Valid example: compute type gpu and no compute requirements",
            "input": {"compute_type": "gpu"},
            "expected_exception": None,
            "expected": {"compute_type": "gpu", "compute_requirements": None},
        },
        {
            "comment": "Valid example: compute type cpu and no compute requirements",
            "input": {"compute_type": "cpu"},
            "expected_exception": None,
            "expected": {"compute_type": "cpu", "compute_requirements": None},
        },
        {
            "comment": "Valid example: compute type gpu and compute requirements",
            "input": {
                "compute_type": "gpu",
                "compute_requirements": ComputeRequirements(min_gpu_vram_mb=4000),
            },
            "expected_exception": None,
            "expected": {
                "compute_type": "gpu",
                "compute_requirements": ComputeRequirements(min_gpu_vram_mb=4000),
            },
        },
        {
            "comment": "Invalid example: compute type cpu and compute requirements",
            "input": {
                "compute_type": "cpu",
                "compute_requirements": ComputeRequirements(min_gpu_vram_mb=4000),
            },
            "expected_exception": ValueError,
        },
    ],
)
def test_compute_type_is_gpu_validator(test):
    expected_exception = test["expected_exception"]
    compute_type = test["input"]["compute_type"]
    compute_requirements = test["input"].get("compute_requirements")
    if expected_exception:
        with pytest.raises(expected_exception):
            PipelineCreate(
                name="pipe",
                variables=[],
                functions=[],
                models=[],
                graph_nodes=[],
                outputs=[],
                compute_type=compute_type,
                compute_requirements=compute_requirements,
            )

    else:
        schema = PipelineCreate(
            name="pipe",
            variables=[],
            functions=[],
            models=[],
            graph_nodes=[],
            outputs=[],
            compute_type=compute_type,
            compute_requirements=compute_requirements,
        )
        assert schema.compute_type == test["expected"]["compute_type"]
        assert schema.compute_requirements == test["expected"]["compute_requirements"]


def test_pipeline_create_to_json():
    schema = PipelineCreate(
        name="pipe",
        variables=[],
        functions=[],
        models=[],
        graph_nodes=[],
        outputs=[],
        compute_type="gpu",
        compute_requirements=ComputeRequirements(min_gpu_vram_mb=4000),
    )
    assert json.loads(schema.json()) == dict(
        name="pipe",
        description="",
        project_id=None,
        public=False,
        tags=[],
        variables=[],
        functions=[],
        models=[],
        graph_nodes=[],
        outputs=[],
        compute_type="gpu",
        compute_requirements=dict(min_gpu_vram_mb=4000),
    )
