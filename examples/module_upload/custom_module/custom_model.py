def generate_a_number() -> int:
    import random

    return random.randint(0, 100)


class MyModel:
    def __init__(self):
        ...

    def random(self) -> int:
        return generate_a_number()
