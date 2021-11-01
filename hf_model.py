from pipeline import Pipeline, Variable, pipeline_function
from pipeline.model.transformer_models import TransformersModel


with Pipeline(pipeline_name="HF pipeline") as pipeline:
    input_str = Variable(variable_type=str, is_input=True)

    hf_model = TransformersModel("EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M")
    output_str = hf_model.predict(input_str)

    pipeline.output(input_str.schema, output_str)

output_pipeline = Pipeline.get_pipeline("HF pipeline")

print(output_pipeline.run("Hello"))
