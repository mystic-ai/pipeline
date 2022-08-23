import dill
from pipeline.objects import Pipeline, Variable

# Test basic Pipeline
def test_with_exit(lol_model):

    with Pipeline("test") as my_pipeline:
        in_1 = Variable(str, is_input=True)
        my_pipeline.add_variable(in_1)

        my_model = lol_model()
        str_1 = my_model.predict(in_1)

        my_pipeline.output(str_1)

    output = Pipeline.run("test", "hey")
    assert output == ["hey lol"]

def test_model_pickle(lol_model):
    with Pipeline("test") as my_pipeline:
        in_1 = Variable(str, is_input=True)
        my_pipeline.add_variable(in_1)

        my_model = lol_model()
        str_1 = my_model.predict(in_1)

        my_pipeline.output(str_1)


    output_pipeline = Pipeline.get_pipeline("test")
    serialized_pipeline = dill.dumps(output_pipeline)
    deserialized_pipeline = dill.loads(serialized_pipeline)
    output = output_pipeline.run("hey")
    assert output == ["hey lol"]
