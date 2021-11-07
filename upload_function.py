from pipeline import pipeline_function

from pipeline.api import upload, authenticate

authenticate("0197246120897461209476120983613409861230986", url="http://localhost:5001")


@pipeline_function
def my_function(vara: float, varb: int) -> str:
    return str(vara + varb)


print(my_function.__function__.__pipeline_function__)
upload(my_function)
