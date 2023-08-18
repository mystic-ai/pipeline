from pipeline import Pipeline, Variable, pipe
from pipeline.cloud.pipelines import upload_pipeline
from pipeline.objects.graph import InputField, InputSchema


class MyInputSchema(InputSchema):
    in_1: int = InputField(lt=5, gt=-5, description="kwarg 1", title="kwarg_1")
    in_2: int | None = InputField(
        default=0, lt=5, gt=-5, description="kwarg 1", title="kwarg_1"
    )


@pipe
def my_func(in_1: int, other_schema: MyInputSchema) -> int:
    return in_1 + other_schema.in_2 + other_schema.in_1


with Pipeline() as builder:
    var_1 = Variable(int, lt=10, ge=0)
    var_2 = Variable(MyInputSchema)

    output = my_func(var_1, var_2)
    builder.output(output)

my_pl = builder.get_pipeline()

rm_pipeline = upload_pipeline(
    my_pl,
    "schema-demo",
    "numpy",
    minimum_cache_number=1,
)
print(rm_pipeline.id)
