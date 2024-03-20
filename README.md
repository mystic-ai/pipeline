# Pipeline SDK [![Version](https://img.shields.io/pypi/v/pipeline-ai)](https://pypi.org/project/pipeline-ai) ![Size](https://img.shields.io/github/repo-size/neuro-ai-dev/pipeline) ![Downloads](https://img.shields.io/pypi/dm/pipeline-ai) [![License](https://img.shields.io/crates/l/ap)](https://www.apache.org/licenses/LICENSE-2.0) [![Discord](https://img.shields.io/badge/discord-join-blue)](https://discord.gg/eJQRkBdEcs)

_Created by [mystic.ai](https://www.mystic.ai/)_

Try premade models for free that have been made using this library: [https://www.mystic.ai/explore](https://www.mystic.ai/explore)

# Table of Contents

- [About](#about)
- [Installation](#installation-and-quickstart)
- [Models](#models)
- [Example and tutorials](#example-and-tutorials)
- [Development](#development)
- [License](#license)

# About

Pipeline is a python library that provides a simple way to construct computational flows for AI/ML models. The library is suitable for both development and production environments supporting inference and training/finetuning. This library is also a direct interface to [Mystic](https://www.mystic.ai/) which provides a compute engine to run pipelines at scale and on enterprise GPUs. This SDK can also be used with [Pipeline Core](https://www.mystic.ai/pipeline-core) on a private hosted cluster.

The syntax used for defining AI/ML pipelines shares some similarities in syntax to sessions in [Tensorflow v1](https://www.tensorflow.org/api_docs/python/tf/compat/v1/InteractiveSession), and Flows found in [Prefect](https://github.com/PrefectHQ/prefect).

# Installation and quickstart

To install pipeline run:

```shell
pip install pipeline-ai
```

To create a new pipeline navigate to the directory you want to create the pipeline in and run:

```shell
pipeline container init -n quickstart
```

This will create two files in the directory:

- `pipeline.yaml` - The configuration file for the container to run the pipeline.
- `new_pipeline.py` - The python file to populate with your pipeline.

# Models

Below are some popular models that have been premade by the community on Mystic. You can find more models in the [explore](https://www.mystic.ai/explore) section of Mystic, and the source code for these models is also referenced in the table.

| Model                                                                                | Category | Description                                            | Source                                                                 |
| ------------------------------------------------------------------------------------ | -------- | ------------------------------------------------------ | ---------------------------------------------------------------------- |
| [meta/llama2-7B](https://www.mystic.ai/meta/llama2-70b)                              | LLM      | A 7B parameter LLM created by Meta (vllm accelerated)  | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |
| [meta/llama2-13B](https://www.mystic.ai/meta/llama2-70b)                             | LLM      | A 13B parameter LLM created by Meta (vllm accelerated) | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |
| [meta/llama2-70B](https://www.mystic.ai/meta/llama2-70b)                             | LLM      | A 70B parameter LLM created by Meta (vllm accelerated) | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |
| [runwayml/stable-diffusion-1.5](https://www.mystic.ai/meta/llama2-70b)               | Vision   | Text -> Image                                          | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |
| [stabilityai/stable-diffusion-xl-refiner-1.0](https://www.mystic.ai/meta/llama2-70b) | Vision   | SDXL Text -> Image                                     | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |
| [matthew/e5_large-v2](https://www.mystic.ai/matthew/e5_large-v2/play)                | LLM      | Text embedding                                         | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |
| [matthew/musicgen_large](https://www.mystic.ai/matthew/musicgen_large/play)          | Audio    | Music generation                                       | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |
| [matthew/blip](https://www.mystic.ai/matthew/blip/play)                              | Vision   | Image captioning                                       | [source](https://github.com/mystic-ai/pipeline/tree/main/examples/nlp) |

# Example and tutorials

| Tutorial                                                                         | Description                                           |
| -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [Entity objects](https://docs.mystic.ai/docs/entity-objects)                     | Use entity objects to persist values and store things |
| [Cold start optimisations](https://docs.mystic.ai/docs/cold-start-optimisations) | Premade functions to do heavy tasks seperately        |
| [Input/output types](https://docs.mystic.ai/docs/inputs-outputs)                 | Defining what goes in and out of your pipes           |
| [Files](https://docs.mystic.ai/docs/files-and-directories)                       | Inputing or outputing files from your runs            |
| [Pipeline building](https://docs.mystic.ai/docs/pipeline-building)               | Building pipelines - how it works                     |
| [Runs](https://docs.mystic.ai/docs/runs)                                         | Running a pipeline remotely - how it works            |

Below is some sample python that demonstrates various features and how to use the Pipeline SDK to create a simple pipeline that can be run locally or on Mystic.

```python
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline

from pipeline import Pipeline, Variable, pipe, entity
from pipeline.cloud import compute_requirements
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema): # TUTORIAL: Input/output types
    height: int | None = InputField(default=512, ge=64, le=1024)
    width: int | None = InputField(default=512, ge=64, le=1024)
    num_inference_steps: int | None = InputField(default=50)
    num_images_per_prompt: int | None = InputField(default=1, ge=1, le=4)
    guidance_scale: int | None = InputField(default=7.5)


@entity # TUTORIAL: Entity objects
class StableDiffusionModel:
    @pipe(on_startup=True, run_once=True) # TUTORIAL: Cold start optimisations
    def load(self):
        model_id = "runwayml/stable-diffusion-v1-5"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
        )
        self.pipe = self.pipe.to(device)

    @pipe
    def predict(self, prompt: str, kwargs: ModelKwargs) -> List[File]: # TUTORIAL: Input/output types
        defaults = kwargs.to_dict()
        images = self.pipe(prompt, **defaults).images

        output_images = []
        for i, image in enumerate(images):
            path = Path(f"/tmp/sd/image-{i}.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(path))
            output_images.append(File(path=path, allow_out_of_context_creation=True)) # TUTORIAL: Files

        return output_images


with Pipeline() as builder: # TUTORIAL: Pipeline building
    prompt = Variable(str)
    kwargs = Variable(ModelKwargs)
    model = StableDiffusionModel()
    model.load()
    output = model.predict(prompt, kwargs)
    builder.output(output)

my_pl = builder.get_pipeline()

```

# Development

This project is made with poetry, [so firstly setup poetry on your machine](https://python-poetry.org/docs/#installation).

Once that is done, please run

```shell
./setup.sh
```

With this you should be good to go. This sets up dependencies, pre-commit hooks and
pre-push hooks.

You can manually run pre commit hooks with

```shell
pre-commit run --all-files
```

To run tests manually please run

```shell
pytest
```

For developing v4, i.e. containerized pipelines, you need to override the installed pipeline-ai python package on the container.
This can be done by bind mounting your target pipeline directory, e.g. using raw docker

# License

Pipeline is licensed under [Apache Software License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
