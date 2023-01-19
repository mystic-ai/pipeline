class NonChargeableProfile(Exception):
    def __init__(self, project_id=None, message="Non chargeable user profile") -> None:
        self.project_id = project_id
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.project_id} -> {self.message}"
