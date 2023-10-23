FROM python:3.10-slim

WORKDIR /app

RUN apt update -y
RUN pip install -U pip

# Install serving packages
RUN pip install -U fastapi==0.103.2 uvicorn==0.15.0 validators==0.22.0

# Container commands
RUN apt update -y
RUN apt install -y git
RUN apt-get update && apt-get -y install gcc mono-mcs && rm -rf /var/lib/apt/lists/*


# Install python dependencies
RUN pip install torch==2.0.1 transformers accelerate==0.21.0 vllm==0.2.0 pandas

# Copy in files
COPY ./ ./
COPY ./examples/docker/ ./


# Remove eventually
RUN pip install ./

ENV PIPELINE_PATH=mistralvllm:my_pipeline

CMD ["uvicorn", "pipeline.container.startup:create_app", "--host", "0.0.0.0", "--port", "80"]
