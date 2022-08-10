from pipeline.objects.variable import Variable


class PipelineFile(Variable):

    path: str

    def __init__(
        self,
        *,
        path: str = None,
        name: str = None,
        remote_id: str = None,
    ) -> None:
        super().__init__(
            type_class=self,
            is_input=False,
            is_output=False,
            name=name,
            remote_id=remote_id,
        )
        self.path = path
