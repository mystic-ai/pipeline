import json

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# tokenized_datasets.set_format("torch")

with open("dataset.jsonl", "w") as jsonl_file:
    for sample in tqdm(tokenized_datasets["train"]):
        jsonl_file.write(f"{json.dumps(sample)}\n")


# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
