from pipeline import Pipeline, Variable
from pipeline.objects.huggingface import TransformersModelForCausalLM

with Pipeline("TransformersModelForCausalLMPipeline") as builder:
    inputs = Variable(
        str, is_input=True
    )  # inputs can actually be List[str] or List[int] too, but pipeline-ai doesn't allow unions
    inference_kwargs = Variable(dict, is_input=True)

    builder.add_variables(inputs, inference_kwargs)

    model = TransformersModelForCausalLM(
        model_path="EleutherAI/gpt-neo-125M",
        tokenizer_path="EleutherAI/gpt-neo-125M",
    )

    output = model.predict(inputs, inference_kwargs)

    builder.output(output)

output_pipeline = Pipeline.get_pipeline("TransformersModelForCausalLMPipeline")

print(
    output_pipeline.run(
        "Hello my name is", {"min_length": 100, "max_length": 150, "temperature": 0.5}
    )
)
