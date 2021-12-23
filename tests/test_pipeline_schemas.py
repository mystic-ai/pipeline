from pydantic.types import SecretBytes
from pipeline.objects import Pipeline, Variable, pipeline_function
from pipeline.objects.graph import Graph
from pipeline.schemas.function import FunctionCreate, FunctionGet
from pipeline.schemas.pipeline import PipelineGet, PipelineVariableGet
from pipeline.schemas.file import FileGet


def test_pipeline_create_schema():
    assert Pipeline._current_pipeline == None

    @pipeline_function
    def add(f_1: float, f_2: float) -> float:
        return f_1 + f_2

    @pipeline_function
    def square(f_1: float) -> float:
        return f_1 ** 2

    assert Pipeline._current_pipeline == None

    with Pipeline("3fq87hiu") as my_pipeline:
        in_1 = Variable(float, is_input=True)
        in_2 = Variable(float, is_input=True)

        add_1 = add(in_1, in_2)
        sq_1 = square(add_1)

        my_pipeline.output(sq_1, add_1)

    graph = Pipeline.get_pipeline("3fq87hiu")
    assert graph.run(3.0, 2.0) == [25.0, 5.0]

    # schema = graph.to_create_schema()
    # assert graph.run(-1.0, 5.0) == [16.0, 4.0]
    # assert len(schema.variables) == 4

    # assert len(schema.functions) == 2
    # assert len(schema) == 2
    # TODO: Add in some more thorough testing


def test_pipeline_get_schema():
    assert Pipeline._current_pipeline == None

    @pipeline_function
    def add(f_1: float, f_2: float) -> float:
        return f_1 + f_2

    @pipeline_function
    def square(f_1: float) -> float:
        return f_1 ** 2

    assert Pipeline._current_pipeline == None

    with Pipeline("3fq87hiu") as my_pipeline:
        in_1 = Variable(float, is_input=True)
        in_2 = Variable(float, is_input=True)

        add_1 = add(in_1, in_2)
        sq_1 = square(add_1)

        my_pipeline.output(sq_1, add_1)

    graph = Pipeline.get_pipeline("3fq87hiu")
    assert graph.run(3.0, 2.0) == [25.0, 5.0]

    # schema = graph.to_create_schema()
    del graph, my_pipeline

    # get_schema = PipelineGet(
    #     id="pipeline-abc123",
    #     name=schema.name,
    #     remote_id="qf34h9uonjl4qf3we",
    #     variables=[
    #         PipelineVariableGet(
    #             remote_id="",
    #             local_id=_var.local_id,
    #             name=_var.name,
    #             type_file=FileGet(
    #                 id="",
    #                 path="./",
    #                 name=".",
    #                 data=_var.type_file.file_bytes,
    #                 file_size=420,
    #             ),
    #             is_input=_var.is_input,
    #             is_output=_var.is_output,
    #         )
    #         for _var in schema.variables
    #     ],
    #     functions=[
    #         FunctionGet(
    #             id="qf39huiojnlqf34r",
    #             name="",
    #             hex_file=FileGet(
    #                 id="",
    #                 path="./",
    #                 name=".",
    #                 data=_func.file.file_bytes,
    #                 file_size=420,
    #             ),
    #             source_sample=_func.function_source,
    #         )
    #         for _func in schema.functions
    #     ],
    #     outputs=schema.outputs,
    #     graph_nodes=schema.graph_nodes,
    # )

    # remade_graph = Graph.from_schema(get_schema)

    # assert remade_graph.run(6.0, 2.0) == [64.0, 8.0]
