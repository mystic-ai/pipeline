from pipeline.objects import File, Graph, Pipeline, Variable, entity, pipe


def onnx_to_pipeline(path: str, name: str = "onnx_model") -> Graph:
    """
    Create a pipeline from an onnx model file
        Parameters:
                path (str): local path to onnx model file
                name (str): Desired name to be given to this pipeline

        Returns:
                pipeline (Graph): Executable Pipeline Graph object
    """

    @entity
    class model:
        def __init__(self):
            self.session = None

        import numpy as np

        @pipe
        def predict(self, onnx_output: list, onnx_input: dict = {}) -> list:
            res = self.session.run(onnx_output, onnx_input)
            return res[0].tolist()

        @pipe(run_once=True, on_startup=True)
        def load(self, onnx_file: File) -> bool:
            import onnxruntime

            self.session = onnxruntime.InferenceSession(
                onnx_file.path,
                providers=[
                    "CUDAExecutionProvider",
                ],
            )
            return True

    with Pipeline(name) as pipeline:
        onnx_file = File(path=path)
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

    return Pipeline.get_pipeline(name)
