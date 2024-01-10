import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects.graph import InputSchema, InputField


HF_MODEL_NAME = "tiiuae/falcon-7b-instruct"
DTYPE = torch.float32

class ModelKwargs(InputSchema):

    temperature: float | None = InputField(default=1.0, le=1.0, gt=0.0)
    top_k: int | None = InputField(default=50)
    top_p: float | None = InputField(default=1.0)
    min_new_tokens: int | None = InputField(default=20)
    max_new_tokens: int | None = InputField(default=100)
    use_cache: bool | None = InputField(default=True)
    do_sample: bool | None = InputField(default=True)
    repetition_penalty: float | None = InputField(default=1.0)

@entity
class FalconPipeline:
    def __init__(self, model_name, dtype) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.dtype = dtype

    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        torch.set_grad_enabled(False)  # Disable gradient calculation globally
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype, trust_remote_code=True
            )
            .cuda()
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    @pipe
    def inference(self, prompt: str, kwargs: ModelKwargs) -> str:
        default_kwargs = {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "min_new_tokens": 20,
            "max_new_tokens": 100,
            "use_cache": True,
            "do_sample": True,
            "repetition_penalty": 1.0,
        }
        tokenizer_ids = dict(eos_token_id = self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
        default_kwargs.update(kwargs)  # Update with user kwargs

        st = time.time()
        in_toks = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        out_toks = self.model.generate(in_toks, **kwargs.to_dict(), **tokenizer_ids)
        output_text = self.tokenizer.decode(
            out_toks[0, len(in_toks[0]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        self.et = time.time() - st

        print(f"Inference function time: {self.et:.2f}s")
        print(f"Token/s: {1/(self.et/out_toks.shape[1]):.3f} tokens")

        return output_text



with Pipeline() as builder:
    # Define the inputs of the pipeline
    prompt = Variable(str)
    kwargs = Variable(ModelKwargs)

    # Main
    _pipeline = FalconPipeline(model_name=HF_MODEL_NAME, dtype=DTYPE)
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    # Define the outputs of the pipeline
    builder.output(out)

my_new_pipeline = builder.get_pipeline()