class PipelineNotDeployed(Exception):
    def __init__(self, pipeline_id=None, message="Pipeline not deployed") -> None:
        self.pipeline_id = pipeline_id
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.pipeline_id} -> {self.message}"
