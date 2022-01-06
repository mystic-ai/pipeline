from pipeline import Pipeline, Variable
from pipeline.model.hf_transformer import TransformersModelForCausalLM

with Pipeline(pipeline_name="HF pipeline") as pipeline:
    input_str = Variable(variable_type=str, is_input=True)

    hf_model = TransformersModelForCausalLM(
        model_path="EleutherAI/gpt-neo-125M",
        tokenizer_path="EleutherAI/gpt-neo-125M",
    )
    output_str = hf_model.predict(input_str)

    pipeline.output(output_str)

output_pipeline = Pipeline.get_pipeline("HF pipeline")

print(output_pipeline.run("Hello"))
