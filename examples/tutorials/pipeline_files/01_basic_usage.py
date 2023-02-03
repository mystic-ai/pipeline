##########
#
#   This example demonstrates how to load a '.pt' file with a Pipeline.
#   It is generaliseable to use a PipelineFile in an arbitary way.
#
##########
import getopt
import sys

import torch

from pipeline import (
    Pipeline,
    PipelineCloud,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)
from pipeline.util.torch_utils import tensor_to_list


@pipeline_model
class MyModel:

    model: torch.nn.Module = None

    def __init__(self):
        self.my_model = torch.nn.Sequential(
            torch.nn.Linear(3, 5), torch.nn.Linear(5, 2)
        )

    @pipeline_function
    def predict(self, x: list[float]) -> str:
        import torch

        # Dimension conversion of x: [3] -> [1, 3]
        assert len(x) == 3, "There must be 3 input numbers in a list"
        x: torch.Tensor = torch.tensor(x).unsqueeze(0)

        return self.my_model(x)

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, model_file: PipelineFile) -> bool:
        import torch

        try:
            print("Loading model...")
            self.my_model.load_state_dict(torch.load(model_file.path))
            self.my_model.eval()
            print("Model loaded!")
        except Exception:
            return False
        return True


with Pipeline("ml-pipeline") as pipeline:
    input_list = Variable(type_class=list, is_input=True)
    model_weight_file = PipelineFile(path="example_weights.pt")

    pipeline.add_variables(input_list, model_weight_file)

    # Note: When the pipeline is uploaded so are the weights.
    # When the Pipeline is loaded on a worker the ".path" variable in the PipelineFile
    # is not the local path any more but a path to the weights on the resource,
    # when the file is loaded on the worker a path is created for it.

    ml_model = MyModel()
    ml_model.load(model_weight_file)

    output = ml_model.predict(input_list)
    output = tensor_to_list(output)
    pipeline.output(output)

output_pipeline = Pipeline.get_pipeline("ml-pipeline")

if __name__ == "__main__":
    argv = sys.argv[1:]

    mode = "run"

    try:
        opts, args = getopt.getopt(argv, "hru", ["run", "upload"])
    except Exception:
        ...

    for opt, arg in opts:
        if opt == "-h":
            print("python model_with_file.py <arg>")
            print("Args:")
            print("-r, --run")
            print("-u, --upload")
        elif opt in ["-r", "--run"]:
            mode = "run"
        elif opt in ["-u", "--upload"]:
            mode = "upload"

    if mode == "run":
        print(output_pipeline.run([2.0, 3.4, 6.0]))
        print(output_pipeline.run([-6.8, 2.1, 1.01]))
    else:
        pc = PipelineCloud()
        uploaded_pipeline = pc.upload_pipeline(output_pipeline)
        output = pc.run_pipeline(uploaded_pipeline, [2.0, 3.4, 6.0])
        print(output.result_preview)
