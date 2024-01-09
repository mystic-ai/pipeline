from pipeline import Pipeline, Variable, entity, pipe


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
    def predict(self, output_number: int) -> str:
        # Perform any operations needed to predict with your model here
        print("Predicting...")

        ...

        print("Prediction complete!")

        return f"Your number: {output_number}"


with Pipeline() as builder:
    input_var = Variable(
        int,
        description="A basic input number to do things with",
        title="Input number",
    )

    my_model = MyModelClass()
    my_model.load()

    output_var = my_model.predict(input_var)

    builder.output(output_var)

my_new_pipeline = builder.get_pipeline()
