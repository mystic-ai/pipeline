# Install Apple library for Stable Diffusion
Install the apple library:

```shell
git clone https://github.com/apple/ml-stable-diffusion.git
python -m pip install -e ./ml-stable-diffusion/
```

Loging to Huggingface
```shell
huggingface-cli login
```

# Convert the model to CoreML
The next step involves converting the Stable Diffusion model to CoreML. When I ran this
it took around 50GB of RAM which is a lot more than your Mac will have (I used a swap
memory work around). I have done the conversion (SD v1.5) and you can do this step by downloading
them directly (OPTION 1) or do it yourself (OPTION 2):

## OPTION 1 - Download the converted files (Recommended)

```bash
wget https://mystic.the-eye.eu/public/AI/sd-apple/sd-apple.tar
tar -xvf sd-apple.tar
```

## OPTION 2 - Convert yourself on a machine with 50GB RAM
_Note: Make sure you have a huggingface token, and accepted terms on the [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) project_


```shell
python -m python_coreml_stable_diffusion.torch2coreml --convert-unet \
--convert-text-encoder --convert-vae-decoder --convert-safety-checker \
--model-version=runwayml/stable-diffusion-v1-5 \
-o .
```

# Run the model

basic testing:
```shell
python -m python_coreml_stable_diffusion.pipeline \
--prompt "a photo of an astronaut riding a horse on mars" \
-i ./ -o testimage.jpg \
--compute-unit ALL --seed 93 --model-version=runwayml/stable-diffusion-v1-5
```

# Create API

Run the python script found in this directory:

```shell
python stable_diffusion_apple.py

```
