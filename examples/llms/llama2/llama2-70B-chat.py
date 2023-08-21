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
    inference: str | None = InputField(
        default="chat_completion", choices=["chat_completion", "stream_inference"]
    )


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
        PATH = "meta-llama/Llama-2-70b-chat-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
            PATH,
            use_auth_token="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",
            torch_dtype=torch.bfloat16,
            device_map="sequential",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            PATH,
            use_auth_token="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",
            use_fast=True,
            device_map="sequential",
        )
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

    def chat_completion(self, dialogs, kwargs):
        import torch

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = kwargs.get(
            "default_system_prompt",
            """You are a helpful, respectful and honest assistant.
                                           Always answer as helpfully as possible, while being safe. Your answers should not include any harmful,
                                           unethical, racist, sexist, toxic, dangerous, or illegal content.
                                           Please ensure that your responses are socially unbiased and positive in nature.
                                           If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
                                           If you don't know the answer to a question, please don't share false information.""",
        )

        prompt_tokens = []
        for dialog in dialogs:
            if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                    }
                ] + dialog
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens = sum(
                [
                    [
                        [self.tokenizer.bos_token_id]
                        + self.tokenizer(
                            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                        ).input_ids
                        + [self.tokenizer.eos_token_id]
                    ]
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += [
                [self.tokenizer.bos_token_id]
                + self.tokenizer(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
                ).input_ids
            ]

            prompt_tokens.append(dialog_tokens)

        generation_tokens = []
        input_num_tokens = []
        for prompt_tok in prompt_tokens:
            # flatten, some inputs may be an array of inputs
            prompt_tok = [[item for sublist in prompt_tok for item in sublist]]
            if kwargs.get("max_new_tokens") == -1:
                kwargs["max_new_tokens"] = self.tokenizer.model_max_length - len(
                    prompt_tok[0]
                )
            generation_tokens.append(
                self.model.generate(torch.tensor(prompt_tok).cuda(), **kwargs)
            )
            input_num_tokens.append(len(prompt_tok[0]))

        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(
                        t[0, input_num_tokens[i] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ),
                }
            }
            for i, t in enumerate(generation_tokens)
        ]

    def stream_inference(self, prompt, kwargs: dict):
        import torch

        if len(prompt) > 1:
            raise ValueError(
                "Found input list of multiple strings. Currently not supported via streaming, \n change to normal inference mode or pass a list with a single string."
            )
        input_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        if kwargs.get("max_new_tokens") == -1:
            kwargs["max_new_tokens"] = self.tokenizer.model_max_length - len(
                input_tokens
            )
        kwargs["input_ids"] = input_tokens
        kwargs["streamer"] = self.streamer
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for text in self.streamer:
            yield text

    def online_inference(self, prompt, kwargs: dict) -> dict:
        import torch

        output = []
        for p in prompt:
            input_tokens = self.tokenizer(p, return_tensors="pt").input_ids.cuda()
            if kwargs.get("max_new_tokens") == -1:
                kwargs["max_new_tokens"] = self.tokenizer.model_max_length - len(
                    input_tokens
                )
            out_toks = self.model.generate(input_tokens, **kwargs)
            output_text = self.tokenizer.decode(
                out_toks[0, len(input_tokens[0]) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            output.append({"result": output_text})
        return output

    @pipe
    def inference(self, prompt: list, kwargs: ModelKwargs):
        default_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        default_kwargs.update(kwargs.to_dict())  # Update with user kwargs

        if default_kwargs["inference"] == "chat_completion":
            del default_kwargs["inference"]
            return self.chat_completion(prompt, default_kwargs)
        elif default_kwargs["inference"] == "streaming":
            del default_kwargs["inference"]
            return self.stream_inference(prompt, default_kwargs)
        else:
            del default_kwargs["inference"]
            return self.online_inference(prompt, default_kwargs)


with Pipeline() as builder:
    prompt = Variable(list)
    kwargs = Variable(ModelKwargs)

    _pipeline = LlamaPipeline()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_pipeline = builder.get_pipeline()

# Upload
result = upload_pipeline(
    my_pipeline,
    "meta/llama2-70B-chat",
    "meta/llama2",
    minimum_cache_number=1,
    required_gpu_vram_mb=150_000,
    accelerators=[
        Accelerator.nvidia_a100_80gb,
        Accelerator.nvidia_a100_80gb,
    ],
)
print(f"Pipeline ID: {result.id}")
