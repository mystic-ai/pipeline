from pipeline import Pipeline, Variable
from pipeline.objects.huggingface import TransformersModelForMaskedLM

with Pipeline("TransformersModelForMaskedLMPipeline") as builder:
    inputs = Variable(
        str, is_input=True
    )  # inputs can actually be List[str] or List[int] too, but pipeline-ai doesn't allow unions
    labels = Variable(
        str, is_input=True
    )  # labels can actually be List[str] or List[int] too, but pipeline-ai doesn't allow unions
    inference_kwargs = Variable(dict, is_input=True)

    builder.add_variables(inputs, labels, inference_kwargs)

    model = TransformersModelForMaskedLM(
        model_path="EleutherAI/gpt-neo-125M",
        tokenizer_path="EleutherAI/gpt-neo-125M",
    )

    output = model.predict(inputs, labels, inference_kwargs)

    builder.output(output)

output_pipeline = Pipeline.get_pipeline("TransformersModelForMaskedLMPipeline")

print(
    output_pipeline.run(
        "The capital of Turkey is <mask>", "The capital of Turkey is Ankara", {}
    )
)
