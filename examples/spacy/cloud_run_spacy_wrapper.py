from pipeline import PipelineCloud, spacy_to_pipeline

api = PipelineCloud(token="pipeline_sk_3NvY0z6XD0i7CIKDZNBhgfn3EwT3_1GH")

# using func arg in spacy_to_pipeline
def func(doc):
    return [[token.text, token.lemma_, token.pos_] for token in doc]


spacy_pipeline = spacy_to_pipeline("en_core_web_sm", func=func, name="spacy get all")


uploaded_pipeline = api.upload_pipeline(spacy_pipeline)
print(f"Uploaded pipeline: {uploaded_pipeline.id}")

print("Run uploaded pipeline")

run_result = api.run_pipeline(
    uploaded_pipeline, ["Apple is looking at buying U.K. startup for $1 billion"]
)
try:
    result_preview = run_result.result_preview
    print("Run result:", result_preview)
except KeyError:
    print(api.download_result(run_result))
