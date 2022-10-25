from pipeline import Pipeline, Variable, pipeline_function, pipeline_model

@pipeline_model
class model:
    def __init__(self):
        ...

    @pipeline_function(run_once=True, on_startup=True)
    def load(self) -> bool:
        import tensorflow_hub as hub

        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return True

    @pipeline_function
    def predict(self, input: list) -> list:
        return self.embed(input)


with Pipeline('tensorflow_universal_sentence_encoder') as pipeline:
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

hf_pipeline = Pipeline.get_pipeline('tensorflow_universal_sentence_encoder')

# example run

input = ["I love Halloween"]

# run locally
[output] = hf_pipeline.run(input)

print(output)
