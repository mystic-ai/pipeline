from pipeline.schemas import *


def test_pipeline_variable_schema():
    test_dict = dict(
        variable_type=str, variable_name="test_schema", is_input=True, is_output=True
    )
    test_schema = PipelineVariableSchema.parse_obj(test_dict)
    output_json = test_schema.json()
    output_dict = test_schema.dict()

    assert output_dict == output_json

    assert output_json["variable_type"] == "str"
    assert output_json["variable_type_file_path"] == None
    assert output_json["variable_name"] == "test_schema"
    assert output_json["is_input"] == True
    assert output_json["is_output"] == True
