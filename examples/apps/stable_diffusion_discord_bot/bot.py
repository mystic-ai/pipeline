import base64
import io
import os
import traceback
import uuid

import interactions

from pipeline.api.asyncio import PipelineCloud

discord_token = os.getenv("DISCORD_TOKEN")
pipeline_token = os.getenv("PIPELINE_API_TOKEN")
bot = interactions.Client(token=discord_token)
pipeline_api = PipelineCloud(token=pipeline_token)


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
    except Exception:
        # If the image generation (or anything else) fails
        # for any reason it's best to let the user know
        await sent_response.edit(
            content="Generation failed, please try again!",
        )

        # With asyncio you have to call the 'flush=True' on print
        print(traceback.format_exc(), flush=True)


bot.start()
