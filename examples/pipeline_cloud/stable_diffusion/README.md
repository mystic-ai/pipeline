# Pipeline AI example: Stable Diffusion inference with custom weights

This example shows how to use your own Stable Diffusion variation, such as Dreambooth, with your own weights. The weights must be in `diffusers` format although there are plenty of conversion scripts out there which can handle converting from `CompVis` style to `diffusers`.

To learn more about Pipeline AI, see [pipeline.ai](https://pipeline.ai).

## Installation

We recommend **Python 3.9** or higher.

**Install with pip**

Install the _pipeline-ai_ library with `pip`:

```console
pip install -U pipeline-ai
```

## Set up your API access

You will need to create an account to use our API.

**Create a Pipeline account**

Create an account via the [Dashboard](https://dashboard.pipeline.ai) then go to "Settings" -> "API Tokens" and create a new token. Copy your API token.

**Add your token to your environment variables**

Copy the API token from your dashboard and add it to your env variables:

```shell
export PIPELINE_API_TOKEN="YOUR_API_TOKEN"
```

## Upload the pipeline with weights

To upload your pipeline, you only need to change one line in `upload.py` before running the script.

Once you've updated the script with your own model, all you need to do is run the script. A pipeline id will be printed to your terminal if everything is successful. Copy this, because you'll need to reference it when submitting runs.

## Run the pipeline

Now, to run the pipeline, all you need to do is add the ID of your uploaded pipeline which came from the previous script or you can find it in your Dashboard.

You can now run this script and it will save the images to a local folder on your computer.

You can look at the `upload.py` script to see what parameters are available, and how to modify your pipeline. By default everything follows the pattern used in our public pipelines, the API for which you can find on the Dashboard -> Pre-trained models -> Stable Diffusion v1.5 txt2img v3.0.

## Help & Support

For anything Pipeline AI related, join our company Discord server for quick support:

[![](https://dcbadge.vercel.app/api/server/7REbAX5v3N)](https://discord.gg/7REbAX5v3N)
