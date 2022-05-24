class InvalidSchema(Exception):
    def __init__(self, schema=None, message="Invalid Schema") -> None:
        self.schema = schema
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.schema} -> {self.message}"
