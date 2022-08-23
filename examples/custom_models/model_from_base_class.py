##########
#
#   This example demonstrates how to load a '.pt' file with a Pipeline.
#   It is generaliseable to use a PipelineFile in an arbitary way.
#
##########
import getopt
import sys
from xml.parsers.expat import model
import dill
import cloudpickle

from pipeline import (
    Pipeline,
    PipelineCloud,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)
from pipeline.objects.decorators import PipelineBase


class MyMatrixModel(PipelineBase):
    @pipeline_function
    def predict(self, x: list[float]) -> list[float]:
        return x

    @pipeline_function(run_once=True, on_startup=True)
    def load(self) -> bool:
        return True

with Pipeline("Matrix pipeline") as pipeline:
    input_list = Variable(type_class=list, is_input=True)

    pipeline.add_variables(input_list)

    # Note: When the pipeline is uploaded so are the weights.
    # When the Pipeline is loaded on a worker the ".path" variable in the PipelineFile
    # is not the local path any more but a path to the weights on the resource,
    # when the file is loaded on the worker a path is created for it.

    matrix_model = MyMatrixModel()

    matrix_model.load()

    output = matrix_model.predict(input_list)
    pipeline.output(output)

output_pipeline = Pipeline.get_pipeline("Matrix pipeline")
# dill.detect.trace(True)
# dill.dumps(output_pipeline)
serialised = dill.dumps(output_pipeline)
serialised = dill.loads(serialised)
print(serialised.__dict__)
print(output_pipeline.__dict__)
breakpoint()
