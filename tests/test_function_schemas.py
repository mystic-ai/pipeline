from pipeline.objects import pipeline_function


def test_function_schema_create():
    @pipeline_function
    def square(f_1: float) -> float:
        return f_1 ** 2

    assert hasattr(square, "__function__") and hasattr(
        square.__function__, "__pipeline_function__"
    )

    # _function: Function = square.__function__.__pipeline_function__

    # schema = _function.to_create_schema()

    assert square(5) == 25.0
    # assert schema.name == "square"
    # assert (
    #    schema.hash
    #    == sha256(inspect.getsource(square.__function__).encode()).hexdigest()
    # )
    # assert schema.function_source == inspect.getsource(square.__function__)

    # assert schema.file_id is None
    # assert schema.file.name == "square"
    # assert schema.file.file_bytes == python_object_to_hex(
    #     square.__function__.__pipeline_function__
    # )


def test_function_get_schema():
    @pipeline_function
    def square(f_1: float) -> float:
        return f_1 ** 2

    # _function: Function = square.__function__.__pipeline_function__
    # schema = _function.to_create_schema()

    del square

    # get_schema = FunctionGet(
    #     id="function_q34f79hiuojnl",
    #     name=schema.name,
    #     hex_file=FileGet(
    #         id="file_d287giuhjk4fq3",
    #         name="square",
    #         path="./",
    #         data=schema.file.file_bytes,
    #         file_size="420",
    #     ),
    #     source_sample=schema.function_source,
    # )

    # function = Function.from_schema(get_schema)
    # square = function.function
    # assert function.typing_inputs["f_1"] == float
    # assert square(3.0) == 9.0
