from pipeline import Pipeline


my_pipeline = Pipeline.load("examples/ML pipeline")
print(my_pipeline.run("Hello"))
