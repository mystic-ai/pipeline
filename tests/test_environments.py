import pytest

from pipeline.cloud.environments import _validate_requirements


def test_requiremetns():
    git_dep = ["git+git+https://github.com/mystic-ai/pipeline"]
    no_version_dep = ["numpy"]

    _validate_requirements(["numpy==1.19.5"])
    _validate_requirements(git_dep)

    with pytest.raises(ValueError):
        _validate_requirements(no_version_dep)

    _validate_requirements(
        [
            "torch==2.0.1",
            "transformers==4.30.2",
            "diffusers==0.19.3",
            "accelerate==0.21.0",
            "hf-transfer~=0.1",
            "vllm==0.1.4",
        ]
    )
