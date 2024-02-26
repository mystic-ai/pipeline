# Generate a 20GB file using numpy

import numpy as np

# Number of 1GB chunks
N = 20

# Create 1GB of random data
data = np.random.bytes(1024**3)

# Write the data to the file
with open("large_file", "wb") as f:
    for _ in range(N):
        f.write(data)
