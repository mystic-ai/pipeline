from pipeline import Pipeline, Variable, pipe


@pipe
def predict(prompt: str) -> str:
    return f"{prompt} lol"


with Pipeline() as builder:
    prompt = Variable(str)
    output = predict(prompt)
    builder.output(output)

pipeline_graph = builder.get_pipeline()
