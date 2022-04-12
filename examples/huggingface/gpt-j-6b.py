from pipeline import (
    Pipeline,
    PipelineCloud,
    Variable,
    pipeline_function,
    pipeline_model,
)


@pipeline_model
class TransformersModelForCausalLM:
    def __init__(
        self,
        model_path: str = "EleutherAI/gpt-neo-125M",
        tokenizer_path: str = "EleutherAI/gpt-neo-125M",
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(self, input_data: str, model_kwargs: dict = {}, **kwargs) -> str:
        import torch

        prompt = str(input_data)
        if len(prompt) < 1:
            return {"error": "Prompt must be a non-empty string."}
        model, tokenizer = self.model, self.tokenizer
        index = 0
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            "cuda:{}".format(index)
        )
        input_token_quantity = torch.numel(input_ids)
        if (
            (
                "response_length" in kwargs
                and kwargs["response_length"] + input_token_quantity > 2048
            )
            or (
                "max_length" in kwargs
                and kwargs["max_length"] + input_token_quantity > 2048
            )
            or (
                "min_length" in kwargs
                and kwargs["min_length"] + input_token_quantity > 2048
            )
        ):
            return {
                "error": (
                    "GPT-J inference is limited to 2048 tokens.",
                    "Reduce the prompt length and/or the expected generation length.",
                )
            }
        if "remove_input" not in kwargs:
            kwargs["remove_input"] = False
        if "penalty" in kwargs:
            kwargs["repetition_penalty"] = kwargs["penalty"]
        if "response_length" in kwargs:
            kwargs["max_length"] = input_token_quantity + kwargs["response_length"]
        if "response_length" in kwargs and "eos_token_id" not in kwargs:
            kwargs["min_length"] = input_token_quantity + kwargs["response_length"]
        if "do_sample" not in kwargs and "num_beams" not in kwargs:
            kwargs["do_sample"] = True

        generation_kwargs = dict(**kwargs, input_ids=input_ids)
        with torch.no_grad():
            outputs = model.generate(
                **generation_kwargs,
            )

        # TODO: Don't redefine output so that it can be cleaned on GPU (del technique)
        if kwargs["remove_input"]:
            outputs = outputs[:, input_ids.shape[1] :]

        if "num_return_sequences" in kwargs:
            return {
                "generated_text": tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
            }

        return {
            "generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)
        }

    @pipeline_function
    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        class no_init:
            def __init__(self, modules=None, use_hf_no_init=True):
                if modules is None:
                    self.modules = [
                        torch.nn.Linear,
                        torch.nn.Embedding,
                        torch.nn.LayerNorm,
                    ]
                self.original = {}
                self.use_hf_no_init = use_hf_no_init

            def __enter__(self):
                if self.use_hf_no_init:
                    transformers.modeling_utils._init_weights = False
                for mod in self.modules:
                    self.original[mod] = getattr(mod, "reset_parameters", None)
                    mod.reset_parameters = lambda x: x

            def __exit__(self, type, value, traceback):
                if self.use_hf_no_init:
                    transformers.modeling_utils._init_weights = True
                for mod in self.modules:
                    setattr(mod, "reset_parameters", self.original[mod])

        with no_init():
            if self.model is None:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        revision="float16",
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
                    .half()
                    .to(0)
                )
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)


api = PipelineCloud()
api.authenticate()

with Pipeline("GPT-J-6B") as builder:
    input_str = Variable(str, is_input=True)
    model_kwargs = Variable(dict, is_input=True)

    builder.add_variables(
        input_str,
        model_kwargs,
    )

    hf_model = TransformersModelForCausalLM(
        model_path="EleutherAI/gpt-j-6B",
        tokenizer_path="EleutherAI/gpt-j-6B",
    )

    output_str = hf_model.predict(
        input_str,
        model_kwargs,
    )

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("GPT-J-6B")

print("Now uploading GPT-J-6B pipeline")
uploaded_pipeline = api.upload_pipeline(output_pipeline)
print(uploaded_pipeline)
print(api.run_pipeline(uploaded_pipeline, ["Hello my name is", {"max_length": 100}]))
