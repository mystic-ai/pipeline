import os
import json

from npu2.object.model import Model, __model_to_dict__
from npu2.pipeline import Pipeline, __pipline_to_dict__


def save_model(pipeline: Pipeline, path: str):
    model_dict = __model_to_dict__(pipeline)

    with open("%s/model.json" % (path), "w") as conf_file:
        conf_file.write(
            json.dumps(
                {
                    "name": pipeline.name,
                    "pipeline_array_length": len(pipeline.pipeline_array),
                }
            )
        )

    for item in range(0):
        for index, pipeline_func in enumerate(pipeline_dict["pipeline_array"]):
            if not os.path.exists("%s/" % (path)):
                os.makedirs("%s/" % (path))

            with open("%s/%u.nf" % (path, index), "wb") as pipline_func_file:
                pipline_func_file.write(pipeline_func)
