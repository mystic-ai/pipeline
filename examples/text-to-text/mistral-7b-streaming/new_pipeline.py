from threading import Thread

import torch
from transformers import AutoTokenizer, MistralForCausalLM, TextIteratorStreamer

from pipeline import Pipeline, entity, pipe
from pipeline.objects.graph import InputField, InputSchema, Variable
from pipeline.objects.variables import Stream


class ModelKwargs(InputSchema):
    do_sample: bool | None = InputField(default=False)
    use_cache: bool | None = InputField(default=True)
    temperature: float | None = InputField(default=0.6)
    top_k: float | None = InputField(default=50)
    top_p: float | None = InputField(default=0.9)
    max_new_tokens: int | None = InputField(default=100, ge=1, le=4096)


class StoppingStreamer(TextIteratorStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_thread(self, thread: Thread):
        self.thread = thread

    def end(self):
        ...


@entity
class Mistral7B:
    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            MistralForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                use_safetensors=True,
                torch_dtype=torch.float16,
                device_map="auto",
            ).half()
            # .to(self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    @pipe
    def inference(self, prompts: str, kwargs: ModelKwargs) -> Stream[str]:
        streamer = TextIteratorStreamer(self.tokenizer)
        # streamer = StoppingStreamer(self.tokenizer)
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)

        generation_kwargs = dict(inputs, streamer=streamer, **kwargs.to_dict())
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        # streamer.set_thread(thread)
        thread.start()

        # process = multiprocessing.Process(
        #     target=self.model.generate, kwargs=generation_kwargs
        # )
        # process.start()

        return Stream(streamer)


with Pipeline() as builder:
    prompt = Variable(list, default=["My name is"])
    kwargs = Variable(ModelKwargs)

    _pipeline = Mistral7B()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_pipeline = builder.get_pipeline()
