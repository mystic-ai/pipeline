from pipeline import PipelineCloud
from pipeline.objects import (
    Graph,
    Pipeline,
    Variable,
    pipeline_function,
    pipeline_model,
)


def spacy_to_pipeline(language_package: str, name: str = "spacy pipeline") -> Graph:
    """
    Create a pipeline with spacy
        Parameters:
                language_package (str): spacy language package
                name (str): Desired name to be given to this pipeline

        Returns:
                pipeline (Graph): Executable Pipeline Graph object
    """

    @pipeline_model
    class model:
        def __init__(self):
            self.nlp = None

        @pipeline_function
        def predict(self, input: str) -> list:
            doc = self.nlp(input)
            # (optional) your spacy code here or you can return entire spacy object
            # to manipulate on a client that has spacy installed.
            res=[]
            for token in doc:
                res.append([token.text, token.pos_, token.dep_])
            return res

        @pipeline_function(run_once=True, on_startup=True)
        def load(self) -> bool:
            import spacy
            spacy.cli.download(language_package)
            self.nlp = spacy.load(language_package)
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

    return Pipeline.get_pipeline(name)


spacy_pipeline = spacy_to_pipeline("en_core_web_sm")

api = PipelineCloud(token="YOUR_TOKEN_HERE")

uploaded_pipeline = api.upload_pipeline(spacy_pipeline)
print(f"Uploaded pipeline: {uploaded_pipeline.id}")

print("Run uploaded pipeline")

run_result = api.run_pipeline(
    uploaded_pipeline, ["Apple is looking at buying U.K. startup for $1 billion"]
)
try:
    result_preview = run_result["result_preview"]
    print("Run result:", result_preview)
except KeyError:
    print(api.download_result(run_result))
