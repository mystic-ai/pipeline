# [Pipeline](https://pipeline.ai) [![Production workflow](https://github.com/neuro-ai-dev/pipeline/actions/workflows/prod-wf.yml/badge.svg?branch=main)](https://github.com/neuro-ai-dev/pipeline/actions/workflows/prod-wf.yml) [![Version](https://img.shields.io/pypi/v/pipeline-ai)](https://pypi.org/project/pipeline-ai) ![Size](https://img.shields.io/github/repo-size/neuro-ai-dev/pipeline) ![Downloads](https://img.shields.io/pypi/dm/pipeline-ai) [![License](https://img.shields.io/crates/l/ap)](https://www.apache.org/licenses/LICENSE-2.0) [![Hiring](https://img.shields.io/badge/hiring-apply%20here-brightgreen)](https://jobs.lever.co/Neuro)

# About

# Installation instructions

## Linux, Mac OS (intel)

## Mac OS (ARM/M1)

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

[_by neuro_](https://getneuro.ai)

# License

Pipeline is licensed under [Apache Software License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
