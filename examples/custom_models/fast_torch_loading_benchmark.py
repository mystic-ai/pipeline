import transformers

import dill
import time

from pipeline.util.torch_utils import extract_tensors, replace_tensors, get_device

before_bert_time = time.time()
bert = transformers.BertModel.from_pretrained("bert-base-uncased")
after_bert_time = time.time()


model_shell, tensors = extract_tensors(bert)

del bert

with open("tmp_shell", "wb") as tmp_file:
    dill.dump(model_shell, tmp_file)


with open("tensors", "wb") as tmp_file:
    dill.dump(tensors, tmp_file)

del model_shell, tensors

# Start loading the model back
before_hdd_load_time = time.time()

with open("tmp_shell", "rb") as tmp_file:
    model_shell = dill.load(tmp_file)


with open("tensors", "rb") as tmp_file:
    tensors = dill.load(tmp_file)

before_assemble_time = time.time()

replace_tensors(model_shell, tensors)

end_time = time.time()

print(f"Time to load from file: {before_assemble_time - before_hdd_load_time}")
print(f"Time to load model with tensors: {end_time - before_assemble_time}")
print(f"Total time: {end_time - before_hdd_load_time}")
print(f"Bert load time: {after_bert_time - before_bert_time}")
