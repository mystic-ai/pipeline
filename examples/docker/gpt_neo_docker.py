"""
Pipline has support for creating FastAPI endpoints for your Pipelines out of the box.
To demonstrate this we'll create a basic GPT-Neo Pipeline with a custom environment and
deploy the docker image.
"""


from pipeline import Pipeline, Variable, docker, pipeline_function, pipeline_model
from pipeline.objects.environment import Environment


@pipeline_model
class PipelineGPTNeo:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(self, input_data: str) -> str:
        if self.model is None:
            from transformers import GPT2Tokenizer, GPTNeoForCausalLM

            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

        input_ids = self.tokenizer(input_data, return_tensors="pt").input_ids

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text


with Pipeline("GPTNeo") as builder:
    in_1 = Variable(str, is_input=True)
    builder.add_variables(in_1)

    gpt_neo = PipelineGPTNeo()

    out_str = gpt_neo.predict(in_1)

    builder.output(out_str)


gpt_neo_pipeline = Pipeline.get_pipeline("GPTNeo")

env = Environment(
    "gptneo-env",
    dependencies=[
        "transformers==4.24.0",
        "torch==1.13.0",
    ],
)

docker.create_pipeline_api([gpt_neo_pipeline], environment=env)
