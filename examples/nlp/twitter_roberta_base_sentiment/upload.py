import numpy as np
import preprocessor
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import upload_pipeline


def softmax(x):
    """Used to convert raw model output scores into probabilities."""
    x_max = np.amax(x, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, keepdims=True)


#: Choose a name for your custom python environment
PYTHON_ENV_NAME = "twitter-roBERTa"
#: Choose a pointer for your pipeline
PIPELINE_POINTER = "twitter-roBERTa-base-sentiment:latest"
#: The HuggingFace source model
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@entity
class RobertaPipeline:
    def __init__(self) -> None:
        self.tokenizer = None
        self.config = None
        self.model = None
        self.device = None

    @pipe(on_startup=True, run_once=True)
    def load(self) -> None:
        """Load the model, tokenizer and config"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_NAME
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        # Used in postprocessing to map IDs to labels
        self.config = AutoConfig.from_pretrained(HF_MODEL_NAME)

    @pipe
    def preprocess(self, raw_text: str) -> str:
        """Preprocesses the input text by filtering out unwanted strings.
        for further details, see https://github.com/s/preprocessor"""
        options = [
            preprocessor.OPT.URL,
            preprocessor.OPT.MENTION,
            preprocessor.OPT.ESCAPE_CHAR,
            preprocessor.OPT.RESERVED,
        ]
        preprocessor.set_options(*options)
        return preprocessor.clean(raw_text)

    @pipe
    def predict(self, input_text: str) -> list[float]:
        """Tokenize the input and feed it to the model"""
        encoded_input = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        output = self.model(**encoded_input)
        # Detatch scores from the computation graph and converted into a numpy array.
        scores = output[0][0].detach().cpu().numpy()
        return scores

    @pipe
    def postprocess(self, scores: list[float]) -> list[dict[str, float]]:
        """The raw scores from the model are passed through the softmax
        function to convert them into probabilities.
        The final output represents the model's confidence for each class
        (positive, negative, neutral)."""
        probablities = softmax(scores)
        ranking = np.argsort(probablities)
        ranking = ranking[::-1]
        result = [
            dict(
                label=self.config.id2label[ranking[i]],
                score=np.round(float(probablities[ranking[i]]), 4),
            )
            for i in range(probablities.shape[0])
        ]
        return result


#: Define the computational graph for the pipeline
with Pipeline() as builder:
    input_text = Variable(
        str,
        title="input_text",
        description="The text that sentiment analysis will be performed on.",
        max_length=512,
    )
    roberta_pipeline = RobertaPipeline()
    roberta_pipeline.load()
    text = roberta_pipeline.preprocess(input_text)
    scores = roberta_pipeline.predict(text)
    output = roberta_pipeline.postprocess(scores)
    builder.output(output)

# Get the computational graph
pipeline_graph = builder.get_pipeline()

if __name__ == "__main__":
    # Create the custom python environment, if it doesn't exist
    try:
        env_id = create_environment(
            name=PYTHON_ENV_NAME,
            python_requirements=[
                "tweet-preprocessor==0.6.0",
                "torch==2.0.1",
                "transformers==4.32.0",
            ],
        )
        print(f"New environment ID = {env_id}")
        print(
            "Environment will be pre-emptively cached on compute resources so please "
            "wait a few mins before using..."
        )
    except Exception:
        print("Environment already exists, using existing environment...")

    # Upload the pipeline to Catalyst
    uploaded_pipeline = upload_pipeline(
        pipeline_graph,
        PIPELINE_POINTER,
        environment_id_or_name=PYTHON_ENV_NAME,
        required_gpu_vram_mb=1500,
        accelerators=[
            Accelerator.nvidia_t4,
        ],
    )
    print(f"Uploaded {PIPELINE_POINTER}, pipeline: {upload_pipeline}")
