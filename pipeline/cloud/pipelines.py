import typing as t

from pipeline.cloud import http
from pipeline.cloud.files import resolve_run_input_file_object
from pipeline.cloud.schemas.runs import ClusterRunResult, RunCreate, RunInput, RunIOType
from pipeline.objects import File
from pipeline.objects.graph import InputSchema


class PipelineRunError(Exception):
    """Error raised when there was an exception raised during a pipeline run"""

    def __init__(
        self,
        exception: str,
        traceback: t.Optional[str],
    ) -> None:
        self.exception = exception
        self.traceback = traceback
        super().__init__(self.exception, self.traceback)

    def __str__(self):
        error_msg = self.exception
        if self.traceback:
            error_msg += f"\n\nFull traceback from run:\n\n{self.traceback}"
        return error_msg


def _data_to_run_input(data: t.Tuple) -> t.List[RunInput]:
    input_array = []

    for item in data:
        input_type = RunIOType.from_object(item)
        if input_type == RunIOType.file or isinstance(item, File):
            input_schema = resolve_run_input_file_object(item)
            input_array.append(input_schema)
            continue
        elif isinstance(item, InputSchema):
            item_dict = item.to_dict()
            output_dict = dict()
            output_dict.update(item_dict)
            for pair_key, pair_value in item_dict.items():
                pair_value_type = RunIOType.from_object(pair_value)
                if pair_value_type == RunIOType.file:
                    new_schema = resolve_run_input_file_object(pair_value)
                    output_dict[pair_key] = new_schema.dict()
                elif pair_value_type == RunIOType.pkl:
                    raise Exception("Generic python objects are not supported yet.")

            input_schema = RunInput(
                type=RunIOType.dictionary,
                value=output_dict,
                file_path=None,
                file_name=None,
                file_url=None,
            )
            input_array.append(input_schema)

            continue
        elif input_type == RunIOType.pkl:
            raise NotImplementedError("Python object input not yet supported")

        input_schema = RunInput(
            type=input_type,
            value=item,
            file_path=None,
            file_name=None,
            file_url=None,
        )
        input_array.append(input_schema)

    return input_array


def _run_pipeline(run_create_schema: RunCreate):
    res = http.post(
        "/v4/runs",
        json_data=run_create_schema.dict(),
        handle_error=True,
    )

    return ClusterRunResult.parse_raw(res.text)


def run_pipeline(
    pipeline: str,
    *data,
):
    run_create_schema = RunCreate(
        pipeline=pipeline,
        inputs=_data_to_run_input(data),
        async_run=False,
    )

    run_get = _run_pipeline(run_create_schema)
    return run_get
