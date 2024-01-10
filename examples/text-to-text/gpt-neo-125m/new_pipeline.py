import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

from pipeline import Pipeline, Variable, entity, pipe


@entity
class PipelineGPTNeo:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    @pipe(run_once=True, on_startup=True)
    def load(self):
        print(
            "Loading GPT-Neo model...",
            flush=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(
            self.device
        )

        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    @pipe
    def predict(self, input_data: str, length: int) -> str:
        input_ids = self.tokenizer(input_data, return_tensors="pt").input_ids.to(
            self.model.device
        )

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=length,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text


with Pipeline() as builder:
    in_1 = Variable(str)
    gen_length = Variable(int)

    gpt_neo = PipelineGPTNeo()
    gpt_neo.load()

    out_str = gpt_neo.predict(in_1, gen_length)

    builder.output(out_str)

my_new_pipeline = builder.get_pipeline()
