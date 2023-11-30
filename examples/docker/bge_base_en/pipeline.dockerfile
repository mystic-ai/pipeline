FROM python:3.10-slim

WORKDIR /app

RUN apt update -y
RUN pip install -U pip

# Install serving packages
RUN pip install -U fastapi==0.103.2 uvicorn==0.15.0 validators==0.22.0

# Container commands
RUN apt update -y
RUN apt install -y git


# Install python dependencies

RUN pip install git+https://github.com/mystic-ai/pipeline.git@ph/just-balls-in-holes FlagEmbedding==1.1.5

# Copy in files
COPY ./ ./

ENV PIPELINE_PATH=bge_base_en:bge_base_en
ENV PIPELINE_NAME=ross/bge-base-en
ENV PIPELINE_IMAGE=ross/bge-base-en

CMD ["uvicorn", "pipeline.container.startup:create_app", "--host", "0.0.0.0", "--port", "14300"]
