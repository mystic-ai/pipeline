import time
from threading import Thread

from pipeline import Pipeline, entity, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.pipelines import upload_pipeline
from pipeline.objects.graph import InputField, InputSchema, Stream, Variable


class ModelKwargs(InputSchema):
    do_sample: bool | None = InputField(default=False)
    use_cache: bool | None = InputField(default=True)
    temperature: float | None = InputField(default=0.6)
    top_k: float | None = InputField(default=50)
    top_p: float | None = InputField(default=0.9)
    max_new_tokens: int | None = InputField(default=100)
    repetition_penalty: float | None = InputField(default=1.0)


@entity
class LlamaPipeline:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

        self.streamer = None

    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TextIteratorStreamer,
        )

        torch.set_grad_enabled(False)  # Disable gradient calculation globally
        PATH = "meta-llama/Llama-2-7b-chat-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
            PATH,
            use_auth_token="",
            torch_dtype=torch.float16,
            device_map="sequential",
        )
        # self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            PATH,
            use_auth_token="",
            use_fast=True,
            device_map="sequential",
        )

    @pipe
    def inference(self, prompt: list, kwargs: ModelKwargs):
        # default_kwargs = {
        #     "eos_token_id": self.tokenizer.eos_token_id,
        #     "pad_token_id": self.tokenizer.pad_token_id,
        # }
        # default_kwargs.update(kwargs.to_dict())
        default_kwargs = kwargs.to_dict()
        output = []
        for p in prompt:
            input_tokens = self.tokenizer(p, return_tensors="pt").input_ids.cuda()
            if default_kwargs.get("max_new_tokens") == -1:
                default_kwargs[
                    "max_new_tokens"
                ] = self.tokenizer.model_max_length - len(input_tokens)
            out_toks = self.model.generate(input_tokens, **default_kwargs)
            output_text = self.tokenizer.decode(
                out_toks[0, len(input_tokens[0]) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            output.append({"result": output_text})

        return output


with Pipeline() as builder:
    prompt = Variable(list)
    kwargs = Variable(ModelKwargs)

    _pipeline = LlamaPipeline()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_pipeline = builder.get_pipeline()

# Upload
# result = upload_pipeline(
#     my_pipeline,
#     "meta/llama2-70B-chat",
#     "meta/llama2",
#     minimum_cache_number=1,
#     required_gpu_vram_mb=150_000,
#     accelerators=[
#         Accelerator.nvidia_a100_80gb,
#         Accelerator.nvidia_a100_80gb,
#     ],
# )
# print(f"Pipeline ID: {result.id}")
my_pipeline.run(
    [
        "Hello, how are you?",
    ],
    ModelKwargs(),
)
start_time = time.time()
for i in range(5):
    output = my_pipeline.run(
        [
            "Hello, how are you?",
        ],
        ModelKwargs(),
    )
end_time = time.time()
print(f"Avg time: {(end_time - start_time)/5.0}")

print(output[0])
