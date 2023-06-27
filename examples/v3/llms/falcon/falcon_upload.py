import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import Pipeline, Variable, pipeline_function, pipeline_model
from pipeline.v3.pipelines import upload_pipeline


@pipeline_model
class FalconPipeline:
    def __init__(self, model_name, dtype) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.dtype = dtype

    @pipeline_function(on_startup=True, run_once=True)
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

    @pipeline_function
    def inference(self, prompt: str, kwargs: dict) -> str:
        default_kwargs = {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "min_new_tokens": 20,
            "max_new_tokens": 100,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.0,
        }
        default_kwargs.update(kwargs)  # Update with user kwargs

        st = time.time()
        in_toks = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        out_toks = self.model.generate(in_toks, **default_kwargs)
        output_text = self.tokenizer.decode(
            out_toks[0, len(in_toks[0]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        self.et = time.time() - st

        print(f"Inference function time: {self.et:.2f}s")
        print(f"Token/s: {1/(self.et/out_toks.shape[1]):.3f} tokens")

        return output_text


_model_name = "tiiuae/falcon-7b-instruct"
_dtype = torch.float32

with Pipeline() as builder:
    # Define the inputs of the pipeline
    prompt = Variable(str, is_input=True)
    kwargs = Variable(dict, is_input=True)
    builder.add_variables(prompt, kwargs)

    # Main
    _pipeline = FalconPipeline(model_name=_model_name, dtype=_dtype)
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    # Define the outputs of the pipeline
    builder.output(out)

local_pipeline = builder.get_pipeline()
uploaded_pipeline = upload_pipeline(
    local_pipeline,
    "mystic/falcon-7b-instruct:standard",
    environment_id_or_name="falcon_7b",
    required_gpu_vram_mb=31000,
)
print(uploaded_pipeline.id)
