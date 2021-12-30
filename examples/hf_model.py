from pipeline import Paiplain, Pipeline, Variable
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

########################################################################
#                 example of proposal for comparisson                  #
########################################################################

pipeline = Paiplain("HF_pipeline")

hf_model = TransformersModelForCausalLM(
    model_path="EleutherAI/gpt-neo-125M",
    tokenizer_path="EleutherAI/gpt-neo-125M",
)

pipeline.set_stages(TransformersModelForCausalLM.predict)
pipeline.model(hf_model)
named_results = pipeline.run("Hello")
print(named_results)
print(pipeline.get_results())
