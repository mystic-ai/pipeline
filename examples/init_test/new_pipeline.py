from typing import Optional

from pydantic import Field

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects.graph import InputField, InputSchema


class MyInputSchema(InputSchema):
    num = InputField(
        int,
        description="A basic input number to do things with",
        title="Input number",
        gt=1,
        lt=100,
    )
    choices = InputField(str, choices=["blue", "green", "red"])


# Put your model inside of the below entity class
@entity
class MyModelClass:
    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        # Perform any operations needed to load your model here
        print("Loading model...")

        ...

        print("Model loaded!")

    @pipe
    def predict(
        self, output_number: int, s: str | None, a: MyInputSchema
    ) -> list[str | int]:
        # Perform any operations needed to predict with your model here
        print("Predicting...")

        ...

        print("Prediction complete!")

        return [f"Your number: {output_number}", output_number] if s is None else [s]


with Pipeline() as builder:
    num = Variable(
        int,
        description="A basic input number to do things with",
        title="Input number",
        gt=1,
        lt=100,
    )
    choices = Variable(str, choices=["blue", "green", "red"])
    s = Variable(str, default="green", choices=["red", "green", "blue"])

    a = Variable(MyInputSchema)

    my_model = MyModelClass()
    my_model.load()

    output_var = my_model.predict(input_var, s, a)

    builder.output(output_var)

my_new_pipeline = builder.get_pipeline()
