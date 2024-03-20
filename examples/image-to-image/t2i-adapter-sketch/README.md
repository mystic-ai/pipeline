# t2i-adapter-sketch

## Description

T2I Adapter is a network providing additional conditioning to stable diffusion. Each t2i checkpoint takes a different type of conditioning as input and is used with a specific base stable diffusion checkpoint.

![Screenshot 2024-03-20 at 14 11 16](https://github.com/mystic-ai/pipeline/assets/30600046/96ea83db-c5ef-4f1b-90e5-c6fb3ce97899)

### Local development

This Mystic pipeline uses a custom dockerfile. To either run or upload this pipeline, you can build the container using docker by running:

```sh
docker build -t sketch-2-img:latest -f pipeline.dockerfile .
```

Then you can run it locally (assuming you have a GPU), by running:

```sh
docker run -p 14300:14300 --gpus all sketch-2-img:latest
```

If you head to `http://localhost:14300/play`, you will see an auto-generated UI to interact with the pipeline. Note, this pipeline requires aprox. 15GB of VRAM. A100-40GB is recommended.

### Upload

Assuming you have authenticated with Mystic and you have a valid api token, you can now upload your pipeline to your account by simply running,

```
pipeline container push
```