from typing import List, Tuple, Any


from pipeline import Pipeline, pipeline_function, Variable
from pipeline.model.transformer_models import TransformersModel

with Pipeline() as pipeline:
    input_str = Variable(variable_type=str, is_input=True)

    hf_model = TransformersModel("EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M")

    output_str = hf_model.predict(input_str)

    pipeline.output(output_str)
    # pipeline.output(token_ids)

output_pipeline = Pipeline.get_pipeline()

print(pipeline.run("Hello"))
