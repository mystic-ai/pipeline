from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects import File


@entity
class MusicgenModel:
    def __init__(self):
        ...

    @pipe(on_startup=True, run_once=True)
    def load(self):
        from audiocraft.models import MusicGen

        self.model = MusicGen.get_pretrained("large")

    @pipe
    def predict(self, prompt: str, duration: int) -> File:
        from audiocraft.data.audio import audio_write

        self.model.set_generation_params(duration=duration)
        descriptions = [prompt]
        wav = self.model.generate(descriptions)

        for idx, one_wav in enumerate(wav):
            file_path = f"/tmp/{idx}"
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write(
                file_path,
                one_wav.cpu(),
                self.model.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
            )

        output_file = File(path=file_path + ".wav", allow_out_of_context_creation=True)
        return output_file


with Pipeline() as builder:
    prompt = Variable(
        str,
        title="Prompt",
        description='Describe the music to be generated, \
        e.g. "rock song with a long guitar solo"',
    )
    duration = Variable(
        int,
        title="Duration",
        description="Length of the music in seconds, \
        generation can take long so keep numbers low",
    )

    model = MusicgenModel()

    model.load()

    output = model.predict(prompt, duration)

    builder.output(output)

my_pl = builder.get_pipeline()

env_name = "musicgen"
try:
    environments.create_environment(
        env_name,
        python_requirements=[
            "torch==2.0.1",
            "audiocraft",
        ],
    )
except Exception:
    pass


pipelines.upload_pipeline(
    my_pl,
    "musicgen_large",
    environment_id_or_name=env_name,
    required_gpu_vram_mb=16_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_a100,
    ],
)
