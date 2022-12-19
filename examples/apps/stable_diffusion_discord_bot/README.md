Stable diffusion discord bot
============================

Overview
--------

This post demonstrates how to create a discord bot to generate images with Stable Diffusion. We will be using the Stable Diffusion API provided by [pipeline.ai](https://pipeline.ai?utm_source=paulcjh&utm_medium=referral&utm_campaign=paulcjh-slack-image-generation-app) to handle the ML compute. The bot will be created in python using the discord interactions library and can run on your laptop.

![](https://paulcjh.com/assets/technical_posts/stable_diffusion_discord_bot/demo.gif)

_Figure 1. Demo of the bot using the `/paint` command._


**If you don't want to make this yourself, you can add the bot I made to your server (for free) [by clicking here](https://discord.com/oauth2/authorize?client_id=1022993363475116082&permissions=2147485696&scope=bot) or checkout the [git repo here](https://github.com/mystic-ai/pipeline/tree/paul/sd-bot/examples/apps/stable_diffusion_discord_bot).**

Contents
--------

*   [Initial setup](#initial_setup)
*   [Making the bot](#making_the_bot)
*   [Docker & docker compose](#docker)

Initial setup [#](#initial_setup)
---------------------------------

### Discord developer setup

Before we can hookup to the Discord API you will first have to create a Discord developer account. Below is a basic set of steps to do this (or you can view a more in-depth guide [here](https://interactionspy.readthedocs.io/en/latest/quickstart.html)).

1.  Sign in to the [discord developer portal](https://discord.com/developers/applications) (you can use your regular discord account)
2.  Click the `New Application` on the top right
3.  Once the Application has been created click on `Bot` on the left menu and click `Add Bot`
4.  To add this bot to your server navigate to `OAuth2/URL Generator` on the left menu and select `bot` on the first scope and `Use Slash Commands` under the `Bot Permissions`. You'll see a `Generated URL` field populated at the bottom of the page, copy and click on it to add it to your server!
5.  Finally, you need to collect your discord bot token under the `Token` tab on your under the `Bot` tab on the left menu.

### Dependencies

There are two main libraries for working with Discord in python:

*   discord.py ([github](https://github.com/Rapptz/discord.py) & [docs](https://discordpy.readthedocs.io/en/stable/index.html))
*   interactions.py ([github](https://github.com/interactions-py/library) & [docs](https://interactionspy.readthedocs.io/en/latest/quickstart.html))

We will be using `interactions.py`, but they are relatively interchangeable. The full list of dependencies can be found here [requirements.txt](/assets/technical_posts/stable_diffusion_discord_bot/requirements.txt). **This post uses python 3.9 and should work for later python versions.**

You can install the dependencies in your python environment using
`pip install -r requirements.txt # OR python -m pip install -r requirements.txt`

Making the bot [#](#making_the_bot)
-----------------------------------

### Generating an image with Stable Diffusion

The core code to generate the image with the [pipeline.ai](https://pipeline.ai?utm_source=paulcjh&utm_medium=referral&utm_campaign=paulcjh-stable-diffusion-discord-bot) library is straight forward as it has native asyncio support for use in production APIs:

```python
async def handle_generation(image_prompt: str) -> io.BytesIO:
    response = await pipeline_api.run_pipeline(
        "stable-diffusion-v1.5-txt2img-fp16@3.0",
        [
            [
                dict(text_in=image_prompt),
            ],
            dict(
                width=512,
                height=512,
                num_inference_steps=25,
                num_samples=1,
            ),
        ],
    )
    image_b64 = response.result_preview[0][0]["images_out"][0]

    image_decoded = base64.decodebytes(image_b64.encode())
    buffer = io.BytesIO(image_decoded)
    new_uid = str(uuid.uuid4())
    buffer.name = new_uid
    return buffer
```

### Bot client

To add a command to your bot with the `interactions.py` library you create an async function with the `@bot.command(...)` decorator:

```python
# For our '/paint' command we only want to take in a single text prompt,
# this is included in the 'options' field below and the name of the option
# is what's used as a key word argument to our function.
@bot.command(
    name="paint",
    description="Paint an image that has never existed...",
    options=[
        interactions.Option(
            name="prompt",
            description="Image generation prompt",
            type=interactions.OptionType.STRING,
            required=True,
        ),
    ],
)
async def paint(ctx: interactions.CommandContext, prompt: str) -> None:
    # You have to send back a response quickly otherwise
    # Discord thinks that the bot has died.
    sent_response = await ctx.send("Generating image...")

    try:
        image_buffer = await handle_generation(prompt)

        # Edit the original message sent to now include the image and the prompt
        await sent_response.edit(
            files=[
                interactions.api.models.misc.File(
                    filename=prompt + ".jpeg", fp=image_buffer
                )
            ],
            content=prompt
            # You can add another argument 'ephemeral=True' to only show the
            # result to the user that sent the request.
        )
    except:
        # If the image generation (or anything else) fails
        # for any reason it's best to let the user know
        await sent_response.edit(
            content="Generation failed, please try again!",
        )

        # With asyncio you have to call the 'flush=True' on print
        print(traceback.format_exc(), flush=True)
```

Finally we need to add a few things to run the bot and complete our script:

```python
import os
import interactions
import traceback
import base64
import io
import uuid

from pipeline.api.asyncio import PipelineCloud

# The token here is the one we collected earlier from the discord bot
discord_token = os.getenv("DISCORD_TOKEN")
pipeline_token = os.getenv("PIPELINE_API_TOKEN")
bot = interactions.Client(token=discord_token)
pipeline_api = PipelineCloud(token=pipeline_token)

# As defined earlier
async def handle_generation(...) -> None:
    ...

# As defined earlier
@bot.command(...)
async def paint(...) -> None:
    ...

bot.start()
```

This code was saved on my system as `bot.py`, and the environment variables can be passed in as follows:

```bash
env DISCORD_TOKEN=... PIPELINE_API_TOKEN=... python bot.py
```

You can now navigate to your discord server and run `/paint`!

Docker & docker compose [#](#docker)
------------------------------------

The bot will run for as long as you have your terminal open, but to run this system continually Docker is a great solution. Docker has a quick start guide [here](https://docs.docker.com/get-started/) but for this post you will only need it installed and not much further understanding.
This section uses the following project directory layout (the `requirements.txt` is the one described above):
```text
project_dir/
    ./bot.py
    ./requirements.txt
    ./Dockerfile
    ./docker-compose.yml
    ./secrets.env
```

All the bot requires to run is the standard python docker image with our secrets and `bot.py` copied into it.
Here is the `Dockerfile` used for the project:

```Docker
FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/

ENV PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./bot.py /code/

CMD ["python", "bot.py"]
```


This can run standalone via the following commands in the project directory:

```bash
sudo docker build . -t stable-diffusion-discord-bot:main
sudo docker run --env DISCORD_TOKEN=... --env PIPELINE_API_TOKEN=... stable-diffusion-discord-bot:main
```
Alternatively, if you'd like to run the Docker image through docker compose you can use the following `docker-compose.yml` file:
```yaml
version: "3.9"
services:
stable-diffusion-discord-bot:
    container_name: stable-diffusion-discord-bot
    image:
    build:
      context: .
      dockerfile: ./Dockerfile
    env_file:
      - secrets.env
```
To run this you will have to populate the secrets.env with the `DISCORD_TOKEN` & `PIPELINE_API_TOKEN` variables. You can run this simply with:

```bash
sudo docker-compose run --build -d
```
