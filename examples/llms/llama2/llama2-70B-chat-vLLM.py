# flake8: noqa
from typing import List

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

from pipeline import Pipeline, entity, pipe
from pipeline.cloud import environments
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration
from pipeline.objects.graph import InputField, InputSchema, Variable

current_configuration.set_debug_mode(True)


class ModelKwargs(InputSchema):
    do_sample: bool | None = InputField(default=False)
    use_cache: bool | None = InputField(default=True)
    temperature: float | None = InputField(default=0.6)
    top_k: float | None = InputField(default=50)
    top_p: float | None = InputField(default=0.9)
    max_new_tokens: int | None = InputField(default=100, ge=1, le=4096)
    presence_penalty: float | None = InputField(default=1.0)
    default_system_prompt: str | None = InputField(
        default="""You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful,
unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information."""
    )


@entity
class LlamaPipeline:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

        self.streamer = None

    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        from pathlib import Path

        model_dir = Path("~/.cache/huggingface/llama2/70b-chat").expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir = str(model_dir)
        snapshot_download(
            "meta-llama/Llama-2-70b-chat-hf",
            local_dir=model_dir,
            token="",
            force_download=True,
            ignore_patterns=["*.safetensors"],
        )
        self.llm = LLM(
            model_dir,
            dtype="bfloat16",
            tensor_parallel_size=2,
        )
        self.tokenizer = self.llm.get_tokenizer()

    @pipe
    def inference(self, dialogs: list, kwargs: ModelKwargs) -> List[str]:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = kwargs.default_system_prompt

        sampling_params = SamplingParams(
            temperature=kwargs.temperature,
            top_p=kwargs.top_p,
            max_tokens=kwargs.max_new_tokens,
            presence_penalty=kwargs.presence_penalty,
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
            prompt_tok = [[item for sublist in prompt_tok for item in sublist]]
            if kwargs.max_new_tokens == -1:
                sampling_params.max_new_tokens = self.tokenizer.model_max_length - len(
                    prompt_tok[0]
                )
            generation_tokens.append(
                self.llm.generate(
                    prompt_token_ids=prompt_tok,
                    sampling_params=sampling_params,
                )
            )
            input_num_tokens.append(len(prompt_tok[0]))

        return [
            {
                "role": "assistant",
                "content": t[0].outputs[0].text,
            }
            for i, t in enumerate(generation_tokens)
        ]


with Pipeline() as builder:
    prompt = Variable(list)
    kwargs = Variable(ModelKwargs)

    _pipeline = LlamaPipeline()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_pipeline = builder.get_pipeline()


environments.create_environment(
    "meta/llama2-vllm-ray",
    python_requirements=[
        "torch==2.0.1",
        "transformers==4.32.0",
        "diffusers==0.19.3",
        "accelerate==0.21.0",
        "hf-transfer~=0.1",
        "vllm==0.1.4",
        "ray==2.6.3",
        "pandas==2.1.0",  # for ray - will say ray is not installed otherwise
    ],
)

# Upload
result = upload_pipeline(
    my_pipeline,
    "meta/llama2-70B-chat",
    "meta/llama2-vllm-ray",
    required_gpu_vram_mb=150_000,
    accelerators=[
        Accelerator.nvidia_a100_80gb,
        Accelerator.nvidia_a100_80gb,
    ],
)

run_pipeline(
    result.id,
    [
        [
            {
                "role": "system",
                "content": "Reply with only a JSON, like {\"category\": \"{categories}\"}. Classify the tweet into one of the following categories: 'Social media', 'Cats', 'Technology', 'Politics', 'Raves', 'Lifestyle'.",
            },
            {
                "role": "user",
                "content": "I love the new Meta Llama 2 model it's really good, yay Zuck!",
            },
        ]
    ],
    {},
)
