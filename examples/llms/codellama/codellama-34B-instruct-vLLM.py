from typing import Optional
import time

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects.graph import InputField, InputSchema
from pipeline.cloud import pipelines, compute_requirements, environments
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(False)

class ModelKwargs(InputSchema):
    # n: Optional[bool] = InputField(1, title="Number of sequences", description="Number of output sequences to return for the given prompt", ge=1)
    # best_of: Optional[int] = InputField(None, title="Best of", description="Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`.")
    presence_penalty: Optional[float] = InputField(0.1, title="Presence Penalty", description="Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.")
    # frequency_penalty: Optional[float] = InputField(0.01, title="Frequency Penalty", description="Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.")
    temperature: Optional[float] = InputField(0.1, title="Temperature", description="Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.")
    top_p: Optional[float] = InputField(0.9, title="Top P", description="Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.", ge=0.0, le=1.0)
    top_k: Optional[int] = InputField(-1, title="Top K", description="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.")
    # use_beam_search: Optional[bool] = InputField(False, title="Use Beam Search", description="Whether to use beam search instead of sampling.")
    # stop: Optional[list[str]] = InputField(None, title="Stop Strings", description="List of strings that stop the generation when they are generated.The returned output will not contain the stop strings.")
    ignore_eos: Optional[bool] = InputField(False, title="Ignore EOS", description="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.")
    # max_tokens: Optional[int] = InputField(16, title="Max Tokens", description="Maximum number of tokens to generate per output sequence.", ge=1, le=16384)
    # logprobs: Optional[int] = InputField(None, title="Logprobs", description="Number of log probabilities to return per output token.")

@entity
class CodeLlamaPipeline:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

        self.streamer = None

    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from vllm import LLM

        dtype = "float16"
        tensor_parallel_size = 1

        model_dir = Path("~/.cache/huggingface/codellama/CodeLlama-34b-instruct-hf").expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir = str(model_dir)
        print("Caching model...")
        snapshot_download(
            "codellama/CodeLlama-34b-Instruct-hf",
            # force_download=True,
            local_dir=model_dir,
            ignore_patterns=["*.safetensors"],
        )
        # print("💥 Model downloaded.")

        # print(f"💥 Loading model with VLLM dtype={dtype} and tensor_parallel_size={tensor_parallel_size}")
        # start = time.time()
        self.llm = LLM(
            model_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        # print(f"💥 Model loaded in {time.time()-start}")

    @pipe
    def inference(self, prompts: list, max_new_tokens: int, kwargs: ModelKwargs) -> list[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            # n = kwargs.n,
            # best_of=kwargs.best_of,
            presence_penalty=kwargs.presence_penalty,
            # frequency_penalty=kwargs.frequency_penalty,
            temperature=kwargs.temperature,
            top_p=kwargs.top_p,
            top_k=kwargs.top_k,
            # use_beam_search=kwargs.use_beam_search,
            # stop=kwargs.stop,
            ignore_eos=kwargs.ignore_eos,
            max_tokens=max_new_tokens,
            # logprobs=kwargs.logprobs
        )

        result = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        result_text = [r.outputs[0].text for r in result]
        return result_text


with Pipeline() as builder:
    prompt = Variable(list, default=["[INST] <<SYS>> You are a genius 10x programmer <</SYS>> How do I install Python packages? [/INST]"], title="List of prompts", description="For greater efficiency, batch your prompts together.")
    max_new_tokens = Variable(int, default=60, title="Max new tokens", description="Maximum number of tokens to generate per output sequence.", ge=1, le=16384)
    kwargs = Variable(ModelKwargs, title="Advanced parameters")

    _pipeline = CodeLlamaPipeline()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, max_new_tokens, kwargs)

    builder.output(out)


my_pl = builder.get_pipeline()


sys_string = "<<SYS>> You are a genius 10x programmer <</SYS>>"

prompts = [
    f"[INST] {sys_string} How do I clone a GitHub repository? [/INST]",
    # f"[INST] {sys_string} How do I install Python packages? [/INST]",
    # f"[INST] {sys_string} What is Kubernetes? [/INST]",
    # f"[INST] {sys_string} How do I deploy a Docker container? [/INST]",
    # f"[INST] {sys_string} How can I optimize SQL queries? [/INST]",
    # f"[INST] {sys_string} What are some best practices for REST API design? [/INST]",
    # f"[INST] {sys_string} How do I use WebSockets in JavaScript? [/INST]",
    # f"[INST] {sys_string} How do I create a virtual environment in Python? [/INST]",
    # f"[INST] {sys_string} How can I implement authentication in Node.js? [/INST]",
    # f"[INST] {sys_string} How do I set up CI/CD pipelines? [/INST]",
    # f"[INST] {sys_string} How do I use async/await in Python? [/INST]",
    # f"[INST] {sys_string} How can I improve code readability? [/INST]",
    # f"[INST] {sys_string} What's the difference between NoSQL and SQL? [/INST]",
    # f"[INST] {sys_string} How do I handle errors in a Go application? [/INST]",
    # f"[INST] {sys_string} What is microservices architecture? Give me as many examples as possible. [/INST]",
    # f"[INST] {sys_string} How do I use GraphQL? [/INST]",
    # f"[INST] {sys_string} How do I configure a load balancer? [/INST]",
    # f"[INST] {sys_string} How can I debug a memory leak? [/INST]",
    # f"[INST] {sys_string} How do I make a POST request using curl? [/INST]",
    # f"[INST] {sys_string} How do I set environment variables in a Linux shell? [/INST]"
]

settings = {"top_k":-1, "top_p":0.9, "temperature": 0.5, "presence_penalty": 1.176}
model_inputs = [prompts, 20, settings]


local = False
if local:
    # Local run
    start = time.time()
    output = my_pl.run(*model_inputs)
    print(f"Total time taken: {time.time() - start}")
    print(output[0][0])
else:
    ##  Upload and Run

    # Upload
    env = environments.create_environment(name="codellama_w_vllm", python_requirements=["torch==2.0.1", "git+https://github.com/huggingface/transformers.git@main", "accelerate==0.23.0", "ray==2.6.3", "pandas==2.1.0", "vllm==0.1.7"])
    upload = pipelines.upload_pipeline(my_pl, "CodeLlama-34b-instruct:latest", "codellama_w_vllm", required_gpu_vram_mb=79_000, accelerators = [compute_requirements.Accelerator.nvidia_a100_80gb])

    # Remote run
    start = time.time()
    output = pipelines.run_pipeline("CodeLlama-34b-instruct:latest", *model_inputs)
    end = time.time() - start
    for i, text in enumerate(output.result.result_array()[0]):
        print(text)
        print("--------------------------")
    
print(f"💥 Total time taken: {end} 💥")
print(f"Total number of processed respones: {i+1}")