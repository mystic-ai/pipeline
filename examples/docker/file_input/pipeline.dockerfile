FROM python:3.10-slim

WORKDIR /app

RUN apt update -y
RUN pip install -U pip

# Install serving packages
RUN pip install -U fastapi==0.103.2 uvicorn==0.15.0 \
    validators==0.22.0 python-multipart==0.0.6


# Container commands
RUN apt update -y
RUN apt install -y git


# Install python dependencies

RUN pip install pipeline-ai==1.0.26

# Copy in files
COPY ./ ./

ENV PIPELINE_PATH=my_pipeline:pipeline_graph
ENV PIPELINE_NAME=plutopulp/file-size-pipeline:latest
ENV PIPELINE_IMAGE=plutopulp/file-size-pipeline:latest

CMD ["uvicorn", "pipeline.container.startup:create_app", "--host", "0.0.0.0", "--port", "14300"]
