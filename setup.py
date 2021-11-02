import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pipeline-ai",
    version="0.0.2",
    author="Paul Hetherington",
    author_email="paul@getneuro.ai",
    description="A simple way to constuct processing pipelines for AI/ML projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuro-ai-dev/pipeline",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "pipeline"},
    packages=setuptools.find_packages(where="pipeline"),
    python_requires=">=3.6",
)
