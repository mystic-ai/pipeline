from pipeline import Pipeline, Variable
from pipeline.objects.huggingface.TransformersModelForCausalLM import (
    TransformersModelForCausalLM,
)

with Pipeline("HF pipeline") as builder:
    input_str = Variable(str, is_input=True)
    model_kwargs = Variable(dict, is_input=True)

    builder.add_variables(input_str, model_kwargs)

    hf_model = TransformersModelForCausalLM(
        model_path="EleutherAI/gpt-neo-125M",
        tokenizer_path="EleutherAI/gpt-neo-125M",
    )
    hf_model.load()
    output_str = hf_model.predict(input_str, model_kwargs)

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("HF pipeline")

print(
    output_pipeline.run(
        "Hello my name is", {"min_length": 100, "max_length": 150, "temperature": 0.5}
    )
)
