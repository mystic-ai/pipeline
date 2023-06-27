import sys

from pipeline.v3.pipelines import stream_pipeline

pointer = "mystic/falcon-7b:streaming"

input_data = [
    """
You are a marketing expert focusing on software products.
You are working on a new product called Mystic and you want
people to know how great it is.
It's really fast, reliable, and scales well.
The product (Mystic) is machine learning (or AI) infrastructure.
Here is a list of 10 tweets from people saying what they love about Mystic:

1. @mystic_ai is the best! We've been using them for years and they are always reliable.
2. I love @mystic_ai #mystic_ai
3. After using @mystic_ai for a few months, I can't imagine using anything else.
4. We managed to get a model up and running in less than 10 minutes using @mystic_ai!
""",
    {
        "repetition_penalty": 1.6,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
    },
]

input_data[0] = "I like doing jiu-jitsu, are there any good spots in london for this?"

for output in stream_pipeline(pointer, *input_data):
    sys.stdout.write(output.value)
    sys.stdout.flush()

print("\nDone!")
