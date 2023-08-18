import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from pipeline import Pipeline, entity, pipe
from pipeline.cloud.pipelines import upload_pipeline
from pipeline.objects.variable import Stream, Variable


@entity
class FalconPipeline:
    def __init__(self, model_name, dtype, load_in_8bit) -> None:
        self.model = None
        self.tokenizer = None

        self.streamer = None

        self.model_name = model_name
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit

    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        torch.set_grad_enabled(False)
        start_time = time.time()
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                load_in_8bit=self.load_in_8bit,
                trust_remote_code=True,
            )
            .cuda()
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, clean_up_tokenization_spaces=True
        )
        self.loading_time = time.time() - start_time

    @pipe
    def streaming_function(self, prompt: str, kwargs: dict) -> Stream[str]:
        st = time.time()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        default_kwargs = {
            "input_ids": input_ids,
            "streamer": self.streamer,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "min_new_tokens": 20,
            "max_new_tokens": 200,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,
        }
        default_kwargs.update(kwargs)  # Update with user kwargs

        thread = Thread(target=self.model.generate, kwargs=default_kwargs)
        thread.start()

        self.et = time.time() - st
        print(f"Inference function time: {self.et:.2f}s")

        for text in self.streamer:
            yield text


_model_name = "tiiuae/falcon-7b"
_dtype_name = "f32"
_dtype = torch.float32

name = f"{_model_name.split('/')[-1]}-{_dtype_name}-streaming"
print(f"Running {name}")

config = {
    "model_name": _model_name,
    "dtype": _dtype,
    "load_in_8bit": [True if _dtype_name == "8bit" else False][0],
}

with Pipeline() as builder:
    prompt = Variable(str, is_input=True)
    kwargs = Variable(dict, is_input=True)
    builder.add_variables(prompt, kwargs)

    _pipeline = FalconPipeline(**config)
    _pipeline.load_model()
    out = _pipeline.streaming_function(prompt, kwargs)

    builder.output(out)


falcon_pipeline = builder.get_pipeline()
new_pipeline = upload_pipeline(
    falcon_pipeline,
    "mystic/falcon-7b:streaming",
    environment_id_or_name="falcon_7b",
    required_gpu_vram_mb=31000,
)
print(new_pipeline)
