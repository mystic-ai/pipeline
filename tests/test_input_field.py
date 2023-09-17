from pipeline.objects.graph import InputField, InputSchema


def test_check_default_value():
    class TestInputSchema(InputSchema):
        input_1: int | None = InputField(default=1)
        input_2: int | None = InputField()

    assert TestInputSchema().input_1 == 1
    assert TestInputSchema(input_1=2).input_1 == 2
    assert TestInputSchema().input_2 is None


# Add test for incorrect input types
