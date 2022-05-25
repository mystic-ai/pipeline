class MissingActiveToken(Exception):
    def __init__(self, token=None, message="Missing Active Token") -> None:
        self.token = token
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.token} -> {self.message}"
