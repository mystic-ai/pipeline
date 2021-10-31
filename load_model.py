from typing import List, Tuple, Any


from pipeline import Pipeline, pipeline_function, Variable
from pipeline.model import pipeline_model


my_pipeline = Pipeline.load("ML pipeline")


output_pipeline = Pipeline.get_pipeline()
pipeline = Pipeline()
print(pipeline.run("Hello"))
