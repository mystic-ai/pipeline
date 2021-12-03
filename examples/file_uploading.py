import os
import string
import random

import numpy as np

from dotenv import load_dotenv

load_dotenv("hidden.env")

from pipeline.api import upload_file, authenticate

my_array = np.zeros(
    (
        random.randint(10, 1000),
        random.randint(10, 100),
        random.randint(10, 100),
    )
)
print("Created array of shape:%s" % str(my_array.shape))

array_file_name = "".join(random.choice(string.ascii_lowercase) for i in range(5))

np.save(array_file_name, my_array)

api_token = os.getenv("TOKEN")
authenticate(api_token)

upload_result = upload_file("%s.npy" % array_file_name, "/default/")
print(upload_result)
os.remove("%s.npy" % array_file_name)
