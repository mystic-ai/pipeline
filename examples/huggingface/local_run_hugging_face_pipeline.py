import typing as t

from pipeline import Pipeline, Variable, pipeline_function, pipeline_model

"""
Example pipline from Hugging Face pipeline abstraction
"""

pipeline_name = "hf_roberta_text_classifier"


@pipeline_model
class model:
    def __init__(self):
        self.pipe = None

    @pipeline_function(run_once=True, on_startup=True)
    def load(self) -> bool:
        from transformers import pipeline

        self.pipe = pipeline(model="roberta-large-mnli")
        return True

    @pipeline_function
    def predict(self, input: list) -> list:
        return self.pipe(input)


with Pipeline(pipeline_name) as pipeline:
    input = Variable(list, is_input=True)

    pipeline.add_variables(
        input,
    )

    model = model()
    model.load()

    output = model.predict(
        input,
    )

    pipeline.output(output)

hf_pipeline = Pipeline.get_pipeline(pipeline_name)

# example run

input = ["I love Halloween"]

# run locally
[output] = hf_pipeline.run(input)

print(output)
