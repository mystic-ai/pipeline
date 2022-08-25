##########
#
#   This example demonstrates how to load a '.pt' file with a Pipeline.
#   It is generaliseable to use a PipelineFile in an arbitary way.
#
##########
import getopt
import sys

import numpy as np

from pipeline import (
    Pipeline,
    PipelineCloud,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)


@pipeline_model
class MyMatrixModel:

    matrix: np.ndarray = None

    def __init__(self):
        ...

    @pipeline_function
    def predict(self, x: list[float]) -> list[float]:
        import numpy as np

        # Dimension conversion of x: [3] -> [1, 3]
        return np.array(np.dot(np.array([x]), self.matrix), dtype=np.float64).tolist()

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, matrix_file: PipelineFile) -> bool:
        try:
            print("Loading matrix...")
            self.matrix = np.load(matrix_file.path)
            print("Loaded matrix!")
        except:
            return False
        return True


np.save("example_matrix.npy", np.random.rand(3, 7))

with Pipeline("Matrix pipeline") as pipeline:
    input_list = Variable(type_class=list, is_input=True)
    matrix_file = PipelineFile(path="example_matrix.npy")

    pipeline.add_variables(input_list, matrix_file)

    # Note: When the pipeline is uploaded so are the weights.
    # When the Pipeline is loaded on a worker the ".path" variable in the PipelineFile
    # is not the local path any more but a path to the weights on the resource,
    # when the file is loaded on the worker a path is created for it.

    matrix_model = MyMatrixModel()
    matrix_model.load(matrix_file)

    output = matrix_model.predict(input_list)
    pipeline.output(output)

output_pipeline = Pipeline.get_pipeline("Matrix pipeline")

if __name__ == "__main__":
    argv = sys.argv[1:]

    mode = "run"
    try:
        opts, args = getopt.getopt(argv, "hru", ["run", "upload"])
    except:
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
        output = pc.run_pipeline(uploaded_pipeline, [[2.0, 3.4, 6.0]])
        print(output["run_state"])
        print(output["result_preview"])
