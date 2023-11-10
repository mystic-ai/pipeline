class RunInputException(Exception):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class RunnableError(Exception):
    """Exception raised during the execution of a pipeline."""

    def __init__(
        self,
        exception: Exception,
        traceback: str | None,
    ) -> None:
        self.exception = exception
        self.traceback = traceback
        super().__init__(self.exception, self.traceback)

    def __str__(self):
        return f"RunnableError({repr(self.exception)})"
