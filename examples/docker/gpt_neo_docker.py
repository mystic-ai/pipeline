from pipeline import docker
from pipeline import Pipeline, Variable, pipeline_model, pipeline_function

from pipeline.objects.environment import Environment

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

@pipeline_model
class PipelineGPTNeo:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(self, input_data: str) -> str:
        if self.model == None:
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
#print(gpt_neo_pipeline.run("Hi, how are you?"))

env = Environment("gptneo-env", dependencies=[
    "transformers",
    "torch",
])

docker.create_pipeline_api([gpt_neo_pipeline], environment=env)