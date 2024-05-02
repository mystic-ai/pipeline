from pipeline.cloud.schemas.runs import RunInput, RunIOType


def test_url_with_spaces():
    url_with_spaces = "http://example.com/some file.png"
    expected_url = "http://example.com/some%20file.png"
    # First instance of RunInput
    run_input1 = RunInput(file_url=url_with_spaces, type=RunIOType.file)
    assert (
        run_input1.file_url == expected_url
    ), "URL with spaces should be encoded correctly in the first instance"

    # Second instance of RunInput using the file_url from the first instance
    run_input2 = RunInput(file_url=run_input1.file_url, type=RunIOType.file)
    assert (
        run_input2.file_url == expected_url
    ), "URL with spaces should be encoded correctly in the second instance"


def test_url_without_spaces():
    url_without_spaces = "http://example.com/file.png"
    run_input = RunInput(file_url=url_without_spaces, type=RunIOType.file)
    assert (
        run_input.file_url == url_without_spaces
    ), "URL without spaces should not be altered"


def test_none_url():
    run_input = RunInput(file_url=None, type=RunIOType.file)
    assert run_input.file_url is None, "None URL should remain None"


def test_nested_run_input_encoding():
    nested_input = {
        "type": "dictionary",
        "value": {
            "file_1": RunInput(
                type="file", file_url="http://example.com/some file.png"
            ),
            "file_2": RunInput(
                type="file", file_url="http://example.com/another file.png"
            ),
        },
        "file_url": None,
    }
    run_input = RunInput(**nested_input)
    assert (
        run_input.value["file_1"].file_url == "http://example.com/some%20file.png"
    ), "Nested URL file_1 should be encoded correctly"
    assert (
        run_input.value["file_2"].file_url == "http://example.com/another%20file.png"
    ), "Nested URL file_2 should be encoded correctly"


def test_deeply_nested_run_input_encoding():
    deeply_nested_input = {
        "type": "dictionary",
        "value": {
            "level_1": {
                "level_2": {
                    "file_3": RunInput(
                        type="file", file_url="http://example.com/yet another file.png"
                    )
                }
            }
        },
        "file_url": None,
    }
    run_input = RunInput(**deeply_nested_input)
    assert (
        run_input.value["level_1"]["level_2"]["file_3"].file_url
        == "http://example.com/yet%20another%20file.png"
    ), "Deeply nested URL should be encoded correctly"


def test_mixed_content_encoding():
    mixed_content_input = {
        "type": "dictionary",
        "value": {
            "file_4": RunInput(
                type="file", file_url="http://example.com/file with space.png"
            ),
            "non_file_data": {
                "file_5": RunInput(
                    type="file",
                    file_url="http://example.com/another file with space.png",
                )
            },
        },
        "file_url": None,
    }
    run_input = RunInput(**mixed_content_input)
    assert (
        run_input.value["file_4"].file_url
        == "http://example.com/file%20with%20space.png"
    ), "Mixed content URL file_4 should be encoded correctly"
    assert (
        run_input.value["non_file_data"]["file_5"].file_url
        == "http://example.com/another%20file%20with%20space.png"
    ), "Mixed content URL file_5 should be encoded correctly"


def test_run_input_list():
    input_list = [
        {
            "type": "file",
            "value": None,
            "file_url": "https://storage.googleapis.com/catalyst-v4/pipeline_files/6f/d2/image 0.jpeg",  # noqa
        },
        {
            "type": "dictionary",
            "value": {
                "file_1": {
                    "type": "file",
                    "value": None,
                    "file_url": "https://storage.googleapis.com/catalyst-v4/pipeline_files/d4/99/image 0.jpeg",  # noqa
                },
                "file_2": {
                    "type": "file",
                    "value": None,
                    "file_url": "https://storage.googleapis.com/catalyst-v4/pipeline_files/c7/81/image 0.jpeg",  # noqa
                },
            },
        },
    ]

    # Convert dictionaries to RunInput instances
    run_inputs = []
    for item in input_list:
        if "file_url" in item:
            run_inputs.append(RunInput(**item))
        else:
            # Create a new dictionary for the value field where each sub-item
            # is converted to RunInput
            modified_value = {k: RunInput(**v) for k, v in item["value"].items()}
            # Create RunInput instance with the modified value
            run_inputs.append(RunInput(type=item["type"], value=modified_value))

    # Check the encoding of URLs
    assert (
        run_inputs[0].file_url
        == "https://storage.googleapis.com/catalyst-v4/pipeline_files/6f/d2/image%200.jpeg"  # noqa
    ), "Top-level URL should be encoded correctly"
    assert (
        run_inputs[1].value["file_1"].file_url
        == "https://storage.googleapis.com/catalyst-v4/pipeline_files/d4/99/image%200.jpeg"  # noqa
    ), "Nested URL file_1 should be encoded correctly"
    assert (
        run_inputs[1].value["file_2"].file_url
        == "https://storage.googleapis.com/catalyst-v4/pipeline_files/c7/81/image%200.jpeg"  # noqa
    ), "Nested URL file_2 should be encoded correctly"
