from pickle import PicklingError

import pytest

from pipeline.objects.graph import Graph
from pipeline.util import python_object_to_hex, hex_to_python_object
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


def test_graph_serialisation(pipeline_graph):
    try:
        deserialized = hex_to_python_object(python_object_to_hex(pipeline_graph))
        assert deserialized.__class__ == pipeline_graph.__class__
        assert deserialized.run("add lol")[0] == "add lol lol"
    except (PicklingError, AssertionError) as e:
        pytest.fail(f"Pipeline Graph does not serialize: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
