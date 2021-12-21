import setuptools

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="pipeline-ai",
    version="0.0.8",
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
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
