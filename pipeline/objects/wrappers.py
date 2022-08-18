from pipeline.objects.decorators import pipeline_function, pipeline_model
from pipeline.objects.pipeline import Pipeline
from pipeline.objects.variable import Variable,PipelineFile

def onnx_model(path: str):
    """
    Create a pipeline from an onnx model file 
        Parameters:
                path (str): local path to onnx model file

        Returns:
                pipeline (PipelineGet): Object representing uploaded pipeline.
    """
    @pipeline_model
    class model:
        def __init__(self):
            self.session = None
        import numpy as np
        @pipeline_function
        def predict(self, onnx_output: list, onnx_input: dict = {}) -> list:
            res = self.session.run(onnx_output, onnx_input)
            return res[0].tolist()

        @pipeline_function(run_once=True, on_startup=True)
        def load(self, onnx_file: PipelineFile) -> bool:
            import onnxruntime
            self.session = onnxruntime.InferenceSession(
                onnx_file.path,
                providers=[
                    "CUDAExecutionProvider",
                ],
            )
            return True

        
    with Pipeline("model") as pipeline:
        onnx_file = PipelineFile(path=path)
        onnx_output = Variable(list, is_input=True)
        onnx_input = Variable(dict, is_input=True)

        pipeline.add_variables(
            onnx_file,
            onnx_output,
            onnx_input,
        )

        model = model()
        model.load(onnx_file)

        output = model.predict(
            onnx_output,
            onnx_input,
        )

        pipeline.output(output)

    return Pipeline.get_pipeline("model")
