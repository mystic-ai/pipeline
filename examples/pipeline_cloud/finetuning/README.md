# Finetuning

## Finetuning pipeline

A typical MVP finetuning pipeline should follow this flow:
1. Arbitrary data object passed in
2. Preprocessing
    - Converting to a dataloader
    - Filtering
3. Training
4. Return a new pipeline (see Note 1)




```python
from pipeline.util import object_to_pfile # converts a python object to PipelineFile

with Pipeline("finetuning-pipe") as builder:

    # Dataset is a hex string from a raw python object
    dataset = Variable(Any, is_input=True)
    ft_params = Variable(dict, is_input=True)

    builder.add_variables(dataset, ft_params)

    my_model = BaseModel()

    pre_proc_output = pre_processing(dataset, ft_params)
    ft_output = my_model.finetune(pre_proc_output, ft_params)

    new_pfile = object_to_pfile(ft_output)

    builder.output(new_pfile)

# Assume that "finetuning-pipe" is run on PipelineCloud and generates a PipelineFile with id:"pipeline_file_1234"

```

To now create an inference pipeline from the finetuned model there are two potential options as shown in the below sections.

### Option 1: Redfine inference pipeline

```python
# The below context manager creates a pipeline with the remote PipelineFile
with Pipeline("sd-pipe") as builder:
    prompts = Variable(list, is_input=True)
    batch_kwargs = Variable(dict, is_input=True)

    model_file = PipelineFile.from_id("pipeline_file_1234", name="model_file")

    builder.add_variables(prompts, batch_kwargs, model_file)

    stable_diff_model = StableDiffusionTxt2ImgModel()
    stable_diff_model.load(model_file)

    output = stable_diff_model.predict(prompts, batch_kwargs)
    builder.output(output)
```

### Option 2: Variable reassignment

This is some proposed new functionality to allow modification of pipelines. Assume that the initial inference pipeline (with ID `pipeline_1234`) has an initial `PipelineFile` variable with name `model_file` that stores the model weights.

```python
from pipeline.objects import Graph

remote_pipeline = Graph.from_id("pipeline_1234")

new_weights = PipelineFile.from_id("pipeline_file_1234")

new_pipeline = remote_pipeline.clone()
new_pipeline.replace_variable(variable_name="model_file", new_weights)
```
