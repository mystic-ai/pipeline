class master_dec:
    def __init__(self, _function=None, *, multiplier=1):
        self.function = _function
        self.multiplier = multiplier

        print("Main call")
        print("function on init -> %s" % self.function)

        print(dir(self.function))

    def __call__(self, *args, **kwargs):
        print("----")
        # print(args)
        print("dif")
        # print(self.function)
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # print(self.function)
            self.function = args[0]
            # Function 2
            return self.__wrapper__
        else:
            # Both 1 + 3 call this
            print(self.function)
            return self.function(*args, **kwargs)

    def __wrapper__(self, *args, **kwargs):
        return self.multiplier * self.function(*args, **kwargs)


"""
@master_dec
def add_numbers1(a, b):
    return a + b


@master_dec(multiplier=3)
def add_numbers2(a, b):
    return a + b
"""


class MyClass:
    def __init__(self) -> None:
        ...

    @master_dec
    def add_numbers3(self, a, b):
        return a + b

    def another_func(self):
        ...


mc = MyClass()
add_numbers3 = mc.add_numbers3

# print(dir(mc.another_func))

# assert add_numbers1(3, 4) == 7
# assert add_numbers2(3, 4) == 7 * 3
assert mc.add_numbers3(3, 4) == 7
