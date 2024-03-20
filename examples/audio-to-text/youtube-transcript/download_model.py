import torch
from transformers import AutoModelForSpeechSeq2Seq

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.save_pretrained("./model_weights")
