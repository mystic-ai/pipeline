Install the apple library:

```shell
git clone https://github.com/apple/ml-stable-diffusion.git
python -m pip install -e ./ml-stable-diffusion/
```

convert SD:

```shell
python -m python_coreml_stable_diffusion.torch2coreml --convert-unet \
--convert-text-encoder --convert-vae-decoder --convert-safety-checker \
--model-version=runwayml/stable-diffusion-v1-5 -o .
```

basic testing:
```shell
python -m python_coreml_stable_diffusion.pipeline \
--prompt "a photo of an astronaut riding a horse on mars" \
-i <output-mlpackages-directory> -o testimage.jpg \
--compute-unit ALL --seed 93
```
