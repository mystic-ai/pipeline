import npu2
from npu2.object import model
from npu2.object.model import Model

class CustomModel(Model):

    def __init__(self, name):
        super().__init__(name=name)
        self.var = 0

    @npu2.function()
    def predict(self, *args, **kwargs):
        print("OVERIDE")
        self.var += args[0]
        return args, kwargs


new_model = CustomModel("Test")

@npu2.function()
def predict(self, *args, **kwargs):
    print("Changed")
    self.var += args[0]
    return args, kwargs

new_model.predict = predict
new_model.update_pipelines()
model_output = new_model.predict_pipeline.run(90, hye="lol")

print(model_output)