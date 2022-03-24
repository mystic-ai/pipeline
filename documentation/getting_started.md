---
title: Getting started
excerpt: A quick setup tutorial for the pipeline python library.
category: 61e6de6d35c07f00106bc18d
---
# Contents

- [About](#about)
- [Installation instructions](#installation-instructions)
  * [Linux, Mac (intel)](#linux--mac--intel-)
  * [Mac (arm/M1)](#mac--arm-m1-)

# About

Pipeline is a python library that provides a simple way to construct computational graphs for AI/ML. The library is suitable for both development and production environments supporting inference and training/finetuning. This library is also a direct interface to [Pipeline.ai](https://pipeline.ai) which provides a compute engine to run pipelines at scale and on enterprise GPUs.

The syntax used for defining AI/ML pipelines shares some similarities in syntax to sessions in [Tensorflow v1](https://www.tensorflow.org/api_docs/python/tf/compat/v1/InteractiveSession), and Flows found in [Prefect](https://github.com/PrefectHQ/prefect). In future releases we will be moving away from this syntax to a C based graph compiler which interprets python directly (and other languages) allowing users of the API to compose graphs in a more native way to the chosen language.

# Installation instructions

## Linux, Mac (intel)

```
pip install -U pipeline-ai
```

## Mac (arm/M1)

Due to the ARM architecture of the M1 core it is necessary to take additional steps to install Pipeline, mostly due to the transformers library. We recoomend running inside of a conda environment as shown below.

1. Make sure Rosetta2 is disabled.
2. From terminal run:

```
xcode-select --install
```

3. Install Miniforge, instructions here: [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge) or follow the below:
   1. Download the Miniforge install script here: [https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
   2. Make the shell executable and run
   ```
   sudo chmod 775 Miniforge3-MacOSX-arm64.sh
   ./Miniforge3-MacOSX-arm64.sh
   ```
4. Create a conda based virtual env and activate:

```
conda create --name pipeline-env python=3.9
conda activate pipeline-env
```

5. Install tensorflow

```
conda install -c apple tensorflow-deps
python -m pip install -U pip
python -m pip install -U tensorflow-macos
python -m pip install -U tensorflow-metal
```

6. Install transformers

```
conda install -c huggingface transformers -y
```

7. Install pipeline

```
python -m pip install -U pipeline-ai
```