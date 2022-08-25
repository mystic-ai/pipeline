import os

from cloudpickle import loads, dumps

from pipeline import (
    pipeline_model,
    pipeline_function,
    Pipeline,
    PipelineCloud,
    PipelineFile,
    Variable,
)

from diffusers import StableDiffusionPipeline

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=os.environ.get("HF_TOKEN")
)


# Write the pipeline to a local file
temp_path = "tmp.pipeline"
with open(temp_path, "wb") as tmp_file:
    tmp_file.write(dumps(pipe))


@pipeline_model
class StableDiffusionModel:

    model = None

    def __init__(self):
        ...

    @pipeline_function
    def predict(self, x: list[str]) -> list[str]:
        import base64
        from io import BytesIO

        results = []
        for _str in x:
            res = self.model(_str)["sample"][0]
            buffered = BytesIO()
            res.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            results.append(img_str)

        return results

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, model_file: PipelineFile) -> bool:
        from cloudpickle import loads

        try:
            print("Loading model...")
            with open(model_file.path,"rb") as tmp_file:
                self.model = loads(tmp_file.read())
            self.model.to(0)
            print("Loaded model!")
        except:
            print("Couldn't load model!")
            return False
        return True


with Pipeline("stable_diffusion_pl") as builder:
    input_strs = Variable(list, is_input=True)
    model_pipeline_file = PipelineFile(path=temp_path)

    builder.add_variables(input_strs, model_pipeline_file)

    stable_diff_model = StableDiffusionModel()
    stable_diff_model.load(model_pipeline_file)

    output = stable_diff_model.predict(input_strs)

    builder.output(output)


test_pipeline = Pipeline.get_pipeline("stable_diffusion_pl")
#test_pipeline.run(["A small cat"])

api = PipelineCloud()
uploaded_pipeline = api.upload_pipeline(test_pipeline)
print(f"Uploaded pipeline id: {uploaded_pipeline.id}")

# Cleanup local dir
os.remove(temp_path)
