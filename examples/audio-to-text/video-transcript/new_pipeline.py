import torch
from moviepy.editor import AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from pipeline import File, Pipeline, Variable, entity, pipe


# Put your model inside of the below entity class
@entity
class MyModelClass:
    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        # Perform any operations needed to load your model here
        print("Loading model...")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "./model_weights"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.pipe = pipe

        print("Model loaded!")

    @pipe
    def predict(self, video: File) -> dict:
        print("Predicting...")

        clip = AudioFileClip(str(video.path))
        clip.write_audiofile("sample.mp3")

        result = self.pipe(
            "sample.mp3",
            return_timestamps=True,
        )

        print("Prediction complete!")

        return result


with Pipeline() as builder:
    input_var = Variable(
        File,
        description="Input video",
        title="Input video",
    )

    my_model = MyModelClass()
    my_model.load()

    output_var = my_model.predict(input_var)

    builder.output(output_var)

my_new_pipeline = builder.get_pipeline()
