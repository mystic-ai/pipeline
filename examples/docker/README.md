# Auto-docker API with Pipeline

Pipeline has support for creating FastAPI endpoints for your Pipelines out of the box. To demonstrate this we'll create a basic GPT-Neo Pipeline with a custom environment and deploy the docker image.

To run the code below (and all auto-docker features in Pipeline) an up-to-date installation of docker, and docker compose, is required.

## API architecture

The auto-docker feature from the Pipeline library extends the base docker image found here: [pipeline-docker](https://github.com/mystic-ai/pipeline-docker).

There are three main Docker images that will be added into the final docker compose that are required to run the API:
1. `pipeline-docker` - The main image containing the API.
2. `postgresql` - This is used for traffic logging, and will be extended to include auth.
3. `redis` - For session data.

## GPT Neo example
### Pipeline creation

We'll create a simple pipeline to take in an input string, and run it through GPT-Neo. A primitive caching method is implementing when on the first run the model and tokenizer are created and stored in the main model class.

```python
from pipeline import Pipeline, Variable, pipeline_model, pipeline_function

@pipeline_model
class GPTNeo:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(self, input_data: str) -> str:
        if self.model == None:
            # Import the transformers modules here so that they'represent when
            # executed in a seperate env post serialisation.
            from transformers import GPTNeoForCausalLM, GPT2Tokenizer
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

        input_ids = self.tokenizer(input_data, return_tensors="pt").input_ids

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text

# Create the computation graph using the Pipeline context manager
with Pipeline("gptneo") as builder:
    in_1 = Variable(str, is_input=True)
    builder.add_variables(in_1)

    gpt_neo = GPTNeo()

    out_str = gpt_neo.predict(in_1)

    builder.output(out_str)

gptneo_pl = Pipeline.get_pipeline("gptneo")
```

### Environment creation
Now that we have created the pipeline we can and create the environment that we'll need to run it.

```python
from pipeline.objects.environment import Environment

# An environment takes in a list of dependencies, in our case we only require transformers and torch.
# Note: You can pass in any generic dependency string (including git repos).
pl_env = Environment(
    "gptneo-env",
    dependencies=[
        "transformers==4.24.0",
        "torch==1.13.0"
    ],
)
```

### Auto Docker
Finally, using the environment and pipeline graph we can create the docker files.

```python
from pipeline import docker

docker.create_pipeline_api([gptneo_pl], environment=pl_env)
```

Running this final command will generate 4 files in your local directory:
1. `docker-compose.yml` - A docker compose file containing environment variables, build info etc.
2. `Dockerfile` - The Dockerfile that is used to create the final container
3. `gptneo.graph` - The serialised graph createed by the Pipeline context manager
4. `requirements.txt` - The python dependencies as defined above (`["transformers==4.24.0","torch==1.13.0"]`)

The docker container creates a FastAPI python environment and loads in your pipeline. The endpoint used is the same as the PipelineCloud endpoint but the host name will be your ip on port 5010:

```shell
http://localhost:5010/v2/run
```

### Using the auto-docker API

Start the docker containers in the background by running:

```shell
sudo docker compose up -d
```

You can now send requests to the API running on your local system:

```shell
curl --request POST \
  --url http://localhost:5010/v2/run \
  --header 'Content-Type: application/json' \
  --data '{"pipeline_id":"gptneo", "data":"Hey there my name is"}'
```

## Not supported

Currently the following Variables are not supported with the auto-docker feature:
- PipelineFile
- Local directories as dependencies for the environment
