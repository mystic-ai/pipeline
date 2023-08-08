from typing import Optional

from pipeline.objects.graph import InputField, InputSchema


# Don't accept pkl / file
class MyInputSchema(InputSchema):
    # in_1: int | None = InputField(1, lt=5, gt=-5, description="kwarg 1")
    in_2: Optional[int] = InputField(default=1, lt=5, gt=-5, description="kwarg 1")


input_base = MyInputSchema()
input_schema = input_base.to_schema()

recreated_schema = InputSchema.from_schema(input_schema)
print(recreated_schema.to_schema())

recreated_schema.parse_dict({"in_2": 10})

print(recreated_schema.in_2)

# @pipeline_function
# def my_function(in_1: int, kwarg_dict: MyInputSchema):
#     return kwarg_dict.in_1


# with Pipeline() as builder:
#     var_1 = Variable(
#         int,
#         lt=10,
#         gt=0,
#         description="Variable 1",
#         title="Variable 1",
#     )

#     var_2 = Variable(MyInputSchema)

#     output = my_function(var_1, var_2)
#     builder.output(output)
