import torch

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, environments, pipelines

LANG_MAP = {
    "Afrikaans": "afr",
    "Amharic": "amh",
    "Modern Standard Arabic": "arb",
    "Moroccan Arabic": "ary",
    "Egyptian Arabic": "arz",
    "Assamese": "asm",
    "North Azerbaijani": "azj",
    "Belarusian": "bel",
    "Bengali": "ben",
    "Bosnian": "bos",
    "Bulgarian": "bul",
    "Catalan": "cat",
    "Cebuano": "ceb",
    "Czech": "ces",
    "Central Kurdish": "ckb",
    "Mandarin Chinese": "cmn",
    "Welsh": "cym",
    "Danish": "dan",
    "German": "deu",
    "Greek": "ell",
    "English": "eng",
    "Estonian": "est",
    "Basque": "eus",
    "Finnish": "fin",
    "French": "fra",
    "West Central Oromo": "gaz",
    "Irish": "gle",
    "Galician": "glg",
    "Gujarati": "guj",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Croatian": "hrv",
    "Hungarian": "hun",
    "Armenian": "hye",
    "Igbo": "ibo",
    "Indonesian": "ind",
    "Icelandic": "isl",
    "Italian": "ita",
    "Javanese": "jav",
    "Japanese": "jpn",
    "Kannada": "kan",
    "Georgian": "kat",
    "Kazakh": "kaz",
    "Halh Mongolian": "khk",
    "Khmer": "khm",
    "Kyrgyz": "kir",
    "Korean": "kor",
    "Lao": "lao",
    "Lithuanian": "lit",
    "Ganda": "lug",
    "Luo": "luo",
    "Standard Latvian": "lvs",
    "Maithili": "mai",
    "Malayalam": "mal",
    "Marathi": "mar",
    "Macedonian": "mkd",
    "Maltese": "mlt",
    "Meitei": "mni",
    "Burmese": "mya",
    "Dutch": "nld",
    "Norwegian Nynorsk": "nno",
    "Norwegian Bokmål": "nob",
    "Nepali": "npi",
    "Nyanja": "nya",
    "Odia": "ory",
    "Punjabi": "pan",
    "Southern Pashto": "pbt",
    "Western Persian": "pes",
    "Polish": "pol",
    "Portuguese": "por",
    "Romanian": "ron",
    "Russian": "rus",
    "Slovak": "slk",
    "Slovenian": "slv",
    "Shona": "sna",
    "Sindhi": "snd",
    "Somali": "som",
    "Spanish": "spa",
    "Serbian": "srp",
    "Swedish": "swe",
    "Swahili": "swh",
    "Tamil": "tam",
    "Telugu": "tel",
    "Tajik": "tgk",
    "Tagalog": "tgl",
    "Thai": "tha",
    "Turkish": "tur",
    "Ukrainian": "ukr",
    "Urdu": "urd",
    "Northern Uzbek": "uzn",
    "Vietnamese": "vie",
    "Yoruba": "yor",
    "Cantonese": "yue",
    "Standard Malay": "zsm",
}


@entity
class StableDiffusionModel:
    def __init__(self):
        ...

    @pipe(on_startup=True, run_once=True)
    def load(self):
        from seamless_communication.models.inference import Translator

        self.translator = Translator(
            "seamlessM4T_large",
            "vocoder_36langs",
            torch.device("cuda:0"),
            torch.float32,
        )

    @pipe
    def predict(self, input_text: str, source_lang: str, target_lang: str) -> str:
        translated_text, _, _ = self.translator.predict(
            input_text,
            "t2tt",
            LANG_MAP[target_lang],
            src_lang=LANG_MAP[source_lang],
        )
        return str(translated_text)


with Pipeline() as builder:
    input_text = Variable(str, title="Prompt")
    source_lang = Variable(str, title="Source Language", choices=list(LANG_MAP.keys()))
    target_lang = Variable(str, title="Target Language", choices=list(LANG_MAP.keys()))

    model = StableDiffusionModel()

    model.load()

    output = model.predict(input_text, source_lang, target_lang)
    builder.output(output)

my_pl = builder.get_pipeline()

env_name = "seamless"
try:
    environments.create_environment(
        env_name,
        python_requirements=[
            "torch==2.0.1",
            "git+https://github.com/facebookresearch/seamless_communication@main",
        ],
    )
except Exception:
    pass


pipelines.upload_pipeline(
    my_pl,
    "seamless_m4t_large",
    environment_id_or_name=env_name,
    required_gpu_vram_mb=20_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_l4,
    ],
)
