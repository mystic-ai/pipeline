FROM python:3.10-slim

WORKDIR /app

RUN apt update -y
RUN pip install -U pip

# Install serving packages
RUN pip install -U fastapi==0.103.2 uvicorn==0.15.0 \
    validators==0.22.0 python-multipart==0.0.6


# Container commands
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gcc && echo ""


# Install python dependencies

RUN pip install git+https://github.com/mystic-ai/pipeline.git@ph/just-balls-in-holes torch==2.0.1 transformers diffusers==0.19.3 accelerate==0.21.0 vllm==0.2.1.post1

# Copy in files
COPY ./ ./

ENV PIPELINE_PATH=new_pipeline:my_pipeline
ENV PIPELINE_NAME=paulcjh/mistral-7b
ENV PIPELINE_IMAGE=paulcjh/mistral-7b

CMD ["uvicorn", "pipeline.container.startup:create_app", "--host", "0.0.0.0", "--port", "14300"]
