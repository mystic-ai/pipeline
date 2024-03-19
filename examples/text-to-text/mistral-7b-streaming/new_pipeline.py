from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import httpx
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, MistralForCausalLM, TextIteratorStreamer

from pipeline import Pipeline, entity, pipe
from pipeline.objects.graph import InputField, InputSchema, Stream, Variable


class ModelKwargs(InputSchema):
    # do_sample: bool | None = InputField(default=False)
    # use_cache: bool | None = InputField(default=True)
    # temperature: float | None = InputField(default=0.6)
    # top_k: float | None = InputField(default=50)
    # top_p: float | None = InputField(default=0.9)
    max_new_tokens: int | None = InputField(default=3000, ge=1, le=4096)


class ChatStreamer(TextIteratorStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = 0

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout, block=True)

        if value == self.stop_signal:
            print("----------STOPPING---------", flush=True)
            raise StopIteration()
        else:
            self.index += 1
            return [{"role": "assistant", "content": value}]

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if True:
            self.token_cache.extend(value.tolist())
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

            if text.endswith((".", "!", "?")):
                printable_text = text[self.print_len :] + " "
                self.token_cache = []
                self.print_len = 0

            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len :]
                self.print_len += len(printable_text)
            else:
                printable_text = text[self.print_len : text.rfind(" ") + 1]
                self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)


@entity
class Mistral7B:
    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            MistralForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                use_safetensors=True,
                torch_dtype=torch.float16,
                device_map="auto",
            ).half()
            # .to(self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    @pipe
    def get_search_results(self, prompts: list[list[dict[str, str]]]) -> list:
        chat = prompts[0]
        search_term = chat[0]["content"]

        search_engine_id = "..."
        search_engine_key = "..."
        response_count = 3
        with httpx.Client() as client:
            response = client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=dict(
                    key=search_engine_key,
                    cx=search_engine_id,
                    q=search_term,
                ),
            )

        response_json = response.json()

        response_items = response_json.get("items", [])
        response_items = response_items[: min(len(response_items), response_count)]

        # raw_html = [self.get_webpage(item["link"]) for item in response_items]
        # Get pages in individual threads
        with ThreadPoolExecutor() as executor:
            raw_html = list(
                executor.map(
                    self.get_webpage, [item["link"] for item in response_items]
                )
            )

        print(raw_html, flush=True)

        page_data = [
            {
                "html": BeautifulSoup(html, "html.parser").get_text(),
                "link": page["link"],
                "title": page["title"],
                "description": page["snippet"],
            }
            for page, html in zip(response_items, raw_html)
        ]
        return page_data

    def get_webpage(self, url: str) -> str:
        with httpx.Client(
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",  # noqa
            },
            follow_redirects=True,
        ) as client:
            response = client.get(url)
            return response.text

    @pipe
    def inference(
        self, prompts: list[list[dict[str, str]]], results: list, kwargs: ModelKwargs
    ) -> Stream[list[dict[str, str]]]:
        streamer = ChatStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
            timeout=20,
        )

        chat = prompts[0]
        # first_question = chat[0]["content"]

        # results = self.get_search_results(first_question)
        system_prompt = "You are an assistant. Be short and concise, don't give unnecessary information, always end a message with a fullstop or questionmark. If required, show code snippets in the response in markdown format (don't forget to include the language reference). Here's some raw webpage text to help you answer the user's question."  # noqa
        for res in results:
            # system_prompt += "\nWebsite:\n" + res[: min(500, len(res))]
            system_prompt += "\n"
            system_prompt += "Title: " + res["title"] + "\n"
            system_prompt += "Link: " + res["link"] + "\n"
            system_prompt += "Page text:\n" + res["html"][: min(500, len(res["html"]))]

        print(system_prompt, flush=True)

        system_prompt = {"role": "system", "content": system_prompt}
        chat.insert(0, system_prompt)

        chat_prompt: str = self.tokenizer.apply_chat_template(chat, tokenize=False)

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        generation_kwargs = dict(inputs, streamer=streamer, **kwargs.to_dict())
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()

        return Stream(streamer)


with Pipeline() as builder:
    prompt = Variable(list, default=[[{"user": "My name is"}]])
    kwargs = Variable(ModelKwargs)

    _pipeline = Mistral7B()
    _pipeline.load_model()

    search_results = _pipeline.get_search_results(prompt)
    out = _pipeline.inference(prompt, search_results, kwargs)

    builder.output(out, search_results)


my_pipeline = builder.get_pipeline()
