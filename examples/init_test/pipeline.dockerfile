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


# Install python dependencies

RUN pip install git+https://github.com/mystic-ai/pipeline.git@ph/just-balls-in-holes

# Copy in files
COPY ./ ./

ENV PIPELINE_PATH=new_pipeline:my_new_pipeline
ENV PIPELINE_NAME=mysticai/matthew_test
ENV PIPELINE_IMAGE=mysticai/matthew_test

CMD ["uvicorn", "pipeline.container.startup:create_app", "--host", "0.0.0.0", "--port", "14300"]
