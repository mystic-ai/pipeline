import os
from typing import Union, Any

from pipeline import Pipeline, PipelineCloud, Variable
from pipeline import pipeline_model, pipeline_function


@pipeline_model
class GPTJ6B_Model:
    def __init__(self):
        self.model_path = "EleutherAI/gpt-j-6B"
        self.tokenizer_path = "EleutherAI/gpt-j-6B"
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def GPTJ6B_Predict(self, input_data: str, model_kwargs: dict = {}) -> str:
        import torch
        import numpy as np
        import io

        if model_kwargs.get("hex_decode"):
            bytesio = io.BytesIO(bytes.fromhex(input_data))
            f = np.load(bytesio, allow_pickle=False)
            a = f["0"]
            a = str(a)
            f.close()
            input_data = a

        prompt = str(input_data)
        if len(prompt) < 1:
            raise ValueError("Prompt must be a non-empty string.")
        model, tokenizer = self.model, self.tokenizer
        index = 0
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(index)
        input_token_quantity = torch.numel(input_ids)
        if (
            (
                "response_length" in model_kwargs
                and model_kwargs["response_length"] + input_token_quantity > 2048
            )
            or (
                "max_length" in model_kwargs
                and model_kwargs["max_length"] + input_token_quantity > 2048
            )
            or (
                "min_length" in model_kwargs
                and model_kwargs["min_length"] + input_token_quantity > 2048
            )
        ):
            return {
                "error": "GPT-J inference is limited to 2048 tokens. Reduce the prompt length and/or the expected generation length."
            }
        if "remove_input" not in model_kwargs:
            model_kwargs["remove_input"] = False
        if "penalty" in model_kwargs:
            model_kwargs["repetition_penalty"] = model_kwargs["penalty"]
        if "response_length" in model_kwargs:
            model_kwargs["min_length"] = (
                input_token_quantity + model_kwargs["response_length"]
            )
            model_kwargs["max_length"] = (
                input_token_quantity + model_kwargs["response_length"]
            )
        if "response_length" in model_kwargs and "eos_token_id" not in model_kwargs:
            model_kwargs["min_length"] = (
                input_token_quantity + model_kwargs["response_length"]
            )
        if "do_sample" not in model_kwargs and "num_beams" not in model_kwargs:
            model_kwargs["do_sample"] = True

        generation_kwargs = dict(**model_kwargs, input_ids=input_ids)
        with torch.no_grad():
            outputs = model.generate(
                **generation_kwargs,
            )

        # TODO: Don't redefine output so that it can be cleaned on GPU (del technique)
        if model_kwargs["remove_input"]:
            outputs = outputs[:, input_ids.shape[1] :]

        if "num_return_sequences" in model_kwargs:
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        import torch

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
api.authenticate(os.environ["TOKEN"])

with Pipeline("GPTJ6B_Pipeline") as builder:
    input_str = Variable(str, is_input=True)
    model_kwargs = Variable(dict, is_input=True)

    builder.add_variables(
        input_str,
        model_kwargs,
    )

    model = GPTJ6B_Model()

    output_str = model.GPTJ6B_Predict(
        input_str,
        model_kwargs,
    )

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("GPTJ6B_Pipeline")
uploaded_pipeline = api.upload_pipeline(output_pipeline)
print(uploaded_pipeline)
