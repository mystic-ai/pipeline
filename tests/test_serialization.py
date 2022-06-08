from pipeline.objects.graph import Graph
from pipeline.schemas.pipeline import PipelineGet


def test_model_serialization(pickled_graph):
    test_schema = PipelineGet.parse_obj(pickled_graph)
    reformed_graph = Graph.from_schema(test_schema)

    # Check basic graph is in correct format
    assert len(reformed_graph.models) == 1
    assert len(reformed_graph.functions) == 1
    assert len(reformed_graph.variables) == 2

    # Check that the only node function has been correctly binded to the model
    assert (
        reformed_graph.nodes[0].function.class_instance
        == reformed_graph.models[0].model
    )

    # TODO Add actual run check
    assert reformed_graph.run("add lol")[0] == "add lol lol"
