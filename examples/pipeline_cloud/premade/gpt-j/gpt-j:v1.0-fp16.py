from pipeline import (
    Pipeline,
    PipelineCloud,
    Variable,
    pipeline_function,
    pipeline_model,
)


@pipeline_model
class GPTJ6B_Model:
    def __init__(self):
        self.model_path = "EleutherAI/gpt-j-6B"
        self.tokenizer_path = "EleutherAI/gpt-j-6B"
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def GPTJ6B_Predict(self, input_data: str, inference_kwargs: dict = {}) -> str:
        import torch

        prompt = str(input_data)
        if len(prompt) < 1:
            raise ValueError("Prompt must be a non-empty string.")
        model, tokenizer = self.model, self.tokenizer
        index = 0
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(index)
        input_token_quantity = torch.numel(input_ids)
        if (
            (
                "response_length" in inference_kwargs
                and inference_kwargs["response_length"] + input_token_quantity > 2048
            )
            or (
                "max_length" in inference_kwargs
                and inference_kwargs["max_length"] + input_token_quantity > 2048
            )
            or (
                "min_length" in inference_kwargs
                and inference_kwargs["min_length"] + input_token_quantity > 2048
            )
        ):
            raise ValueError(
                "GPT-J inference is limited to 2048 tokens. Reduce the prompt length"
                " and/or the expected generation length."
            )
        if "include_input" not in inference_kwargs:
            inference_kwargs["include_input"] = False
        if "remove_input" in inference_kwargs:
            inference_kwargs["include_input"] = not inference_kwargs[
                "remove_input"
            ]  # legacy
        if "penalty" in inference_kwargs:
            inference_kwargs["repetition_penalty"] = inference_kwargs["penalty"]
        if "response_length" in inference_kwargs:
            inference_kwargs["min_length"] = (
                input_token_quantity + inference_kwargs["response_length"]
            )
            inference_kwargs["max_length"] = (
                input_token_quantity + inference_kwargs["response_length"]
            )
        if (
            "response_length" in inference_kwargs
            and "eos_token_id" not in inference_kwargs
        ):
            inference_kwargs["min_length"] = (
                input_token_quantity + inference_kwargs["response_length"]
            )
        if "do_sample" not in inference_kwargs and "num_beams" not in inference_kwargs:
            inference_kwargs["do_sample"] = True
        if "pad_token_id" not in inference_kwargs:
            inference_kwargs["pad_token_id"] = tokenizer.eos_token_id

        banned_kwargs = [
            "attention_mask",
            "return_dict_in_generate",
            "output_attentions",
            "output_hidden_states",
            "output_scores",
        ]
        for k in banned_kwargs:
            if k in inference_kwargs:
                raise ValueError(
                    f"Sorry, {k} is not yet a supported parameter."
                    " Let us know if you want it added!"
                )

        inference_kwargs = dict(**inference_kwargs, input_ids=input_ids)
        with torch.no_grad():
            outputs = model.generate(
                **inference_kwargs,
            )

        # TODO: Don't redefine output so that it can be cleaned on GPU (del technique)
        if not inference_kwargs["include_input"]:
            outputs = outputs[:, input_ids.shape[1] :]

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
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


api = PipelineCloud()

with Pipeline("GPT-J") as builder:
    input_str = Variable(str, is_input=True)
    inference_kwargs = Variable(dict, is_input=True)

    builder.add_variables(
        input_str,
        inference_kwargs,
    )

    model = GPTJ6B_Model()

    output_str = model.GPTJ6B_Predict(
        input_str,
        inference_kwargs,
    )

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("GPT-J")
uploaded_pipeline = api.upload_pipeline(output_pipeline)
