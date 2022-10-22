from cloudpickle import dumps

from torch import nn

from pipeline.objects import (
    Graph,
    Pipeline,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)

__all__ = ["torch_to_pipeline"]


def torch_to_pipeline(pytorch_model: nn.Module, name: str = "PyTorch Model") -> Graph:
    """
    Create a pipeline from a PyTorch nn.Module object (a PyTorch model).

    Args:
        pytorch_model (nn.Module): Input model
        name (str, optional): Name for the pipeline. Defaults to "PyTorch Model".

    Returns:
        Graph: The output pipeline graph
    """

    assert isinstance(pytorch_model, nn.Module), "the model passed to 'torch_to_pipeline' must be of type 'nn.Module'"

    # Serialise the module to a local path

    @pipeline_model
    class model:
        def __init__(self):
            self.model: nn.Module = None

        @pipeline_function
        def predict(self):
            ...

        @pipeline_function
        def load(self):
            ...
    

    with Pipeline(name) as builder:
        torch_path = PipelineFile(...)

    return None
