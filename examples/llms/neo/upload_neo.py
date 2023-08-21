from httpx import Response

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import upload_pipeline


@entity
class PipelineGPTNeo:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    @pipe(run_once=True, on_startup=True)
    def load(self):
        import torch
        from transformers import GPT2Tokenizer, GPTNeoForCausalLM

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
    in_1 = Variable(str, is_input=True)
    gen_length = Variable(int, is_input=True)
    builder.add_variables(in_1, gen_length)

    gpt_neo = PipelineGPTNeo()
    gpt_neo.load()

    out_str = gpt_neo.predict(in_1, gen_length)

    builder.output(out_str)

neo_env_name = "neo"

try:
    env_id = create_environment(
        name=neo_env_name,
        python_requirements=[
            "torch",
            "transformers",
        ],
    )
    print(f"New environment ID = {env_id}")
    print(
        "Environment will be pre-emptively cached on compute resources so please "
        "wait a few mins before using..."
    )
except Exception:
    print("Environment already exists, using existing environment...")

gpt_neo_pipeline = builder.get_pipeline()
upload_resonse: Response = upload_pipeline(
    gpt_neo_pipeline,
    "mystic/neo:main",
    environment_id_or_name=neo_env_name,
    required_gpu_vram_mb=1500,
    accelerators=[
        Accelerator.nvidia_t4,
    ],
)
print(f"Uploaded GPTNeo, server response: {upload_resonse}")
