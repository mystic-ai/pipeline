import typing as t

from pipeline.objects import (
    Graph,
    Pipeline,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)


def onnx_to_pipeline(path: str, name: str = "ONNX model") -> Graph:
    """
    Create a pipeline from an onnx model file
        Parameters:
                path (str): local path to onnx model file
                name (str): Desired name to be given to this pipeline

        Returns:
                pipeline (Graph): Executable Pipeline Graph object
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

    with Pipeline(name) as pipeline:
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

    return Pipeline.get_pipeline(name)


def spacy_to_pipeline(
    spacy_model: str, func: t.Optional[t.Callable] = None, name: str = "Spacy pipeline"
) -> Graph:
    """
    Create a pipeline using Spacy
        Parameters:
                spacy_model (str): tokenizer model name (trained Spacy "pipeline")
                func (Optional[Callable]): function to be called on spacy output
                name (str): Name to be given to this pipeline

        Returns:
                pipeline (Graph): Executable Pipeline Graph object
    """

    @pipeline_model
    class model:
        def __init__(self):
            self.nlp = None
            self.func = func

        @pipeline_function
        def predict(self, input: str) -> list:
            doc = self.nlp(input)

            if self.func:
                return self.func(doc)
            return doc

        @pipeline_function(run_once=True, on_startup=True)
        def load(self) -> bool:
            import spacy

            spacy.require_gpu()
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            return True

    with Pipeline(name) as pipeline:
        input = Variable(str, is_input=True)

        pipeline.add_variables(
            input,
        )

        model = model()
        model.load()

        output = model.predict(
            input,
        )

        pipeline.output(output)

    spacy_pipeline = Pipeline.get_pipeline(name)

    return spacy_pipeline
