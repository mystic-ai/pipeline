import base64
import os

from pipeline import PipelineCloud

api = PipelineCloud()
PIPELINE_ID = "UPDATE_WITH_YOUR_PIPELINE_ID"

# Run API inference
run = api.run_pipeline(
    PIPELINE_ID,
    [
        [{"text_in": "An old black cat on a brown leather sofa."}],
        {"num_samples": 1},
    ],
)

# Handle error if any
if run.error_info:
    print(run.error_info.exception)
    quit()


# Create folder to store output images
folder_name = "outputs"
os.makedirs(folder_name, exist_ok=True)
for index, result in enumerate(run.result_preview[0]):
    for sample in result["images_out"]:
        with open(
            f"{folder_name}/sample-{len(os.listdir(folder_name))}.png",
            "wb",
        ) as file:
            file.write(base64.b64decode(sample))
    print(result["metadata"])
