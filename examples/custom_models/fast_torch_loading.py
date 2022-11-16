import torch
import transformers

from typing import List, Dict

from pipeline import Pipeline, pipeline_model, pipeline_function
from pipeline.objects import PipelineFile, Variable
from pipeline.util.torch_utils import extract_tensors

model = transformers.GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

empty_model, model_tensors = extract_tensors(model)


@pipeline_model
class GPTJModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def replace_tensors(m: torch.nn.Module, tensors: List[Dict]) -> None:
        import torch

        modules = [module for _, module in m.named_modules()]
        for module, tensor_dict in zip(modules, tensors):
            for name, array in tensor_dict["params"].items():
                module.register_parameter(
                    name, torch.nn.Parameter(torch.as_tensor(array))
                )
            for name, array in tensor_dict["buffers"].items():
                module.register_buffer(name, torch.as_tensor(array))

    @pipeline_function
    def predict(self, input_str: str) -> str:
        import torch

        prompt = str(input_str)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=25,
                min_length=25,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, model_file: PipelineFile, tensor_file: PipelineFile) -> None:
        import dill
        from transformers import AutoTokenizer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(model_file.path, "rb") as tmp_file:
            self.model = dill.load(tmp_file)

        with open(tensor_file.path, "rb") as tmp_file:
            self.model_tensors = dill.load(tmp_file)

        self.replace_tensors(model, model_tensors)

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


with Pipeline("fast-torch-gptj") as builder:
    input_str = Variable(type_class=str, is_input=True, name="input-prompt")

    model_file = PipelineFile.from_object(empty_model)
    tensors_file = PipelineFile.from_object(empty_model)

    pl_model = GPTJModel()
    pl_model.load(model_file, tensors_file)

    output = pl_model.predict(input_str)

    builder.output(output)
