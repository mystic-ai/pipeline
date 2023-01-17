class NonChargeableProfile(Exception):
    def __init__(self, message: str = "Non chargeable user profile") -> None:
        super().__init__(message)
