import typing as t


class EnvironmentInitializationError(Exception):
    """Error raised if environment could not be initialized successfully"""

    def __init__(self, message: str, traceback: t.Optional[str] = None):
        self.message = message
        self.traceback = traceback
        super().__init(self.message, self.traceback)

    def __str__(self) -> str:
        return self.message
