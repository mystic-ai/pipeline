from pipeline.objects.environment import Environment

# worker_environment = Environment(
#     environment_name="default-worker-environment",
#     dependencies=[
#         "loguru==0.5.3",
#         "celery==5.2.3",
#         "fastapi==0.79.1",
#         "starlette-context==0.3.3",
#         "starlette-exporter==0.12.0",
#         "httpx==0.23.0",
#         "alembic==1.7.6",
#         "deepspeed==0.5.10",
#         "asyncpg-listen==0.0.4",
#         "cloudpickle==2.1.0",
#         "seaborn==0.11.2",
#         "numpy==1.21.0",
#         "tensorflow==2.9.1",
#         "tensorflow-hub==0.12.0",
#         "Pillow==9.1.1",
#         "opencv-python==<=4.5.5",
#         "pydantic==1.9.1",
#         "onnxruntime-gpu==1.13.1",
#         "spacy==3.4.1",
#         "accelerate==0.10.0",
#         "transformers==4.21.2",
#         "diffusers==0.6.0",
#         "einops==0.4.1",
#         "xgboost==1.6.2",
#         "wandb==0.13.3",
#         "setuptools==65.4.1",
#         "scikit-learn==1.1.2",
#         "sentence-transformers==2.2.2",
#         "catboost==1.1",
#         "arize==5.0.3",
#     ],
# )

base_worker_environment = Environment(
    environment_name="base-worker-environment",
    dependencies=[
        "fastapi==0.79.1",
        "starlette-context==0.3.3",
        "starlette-exporter==0.12.0",
        "httpx==0.23.1",
        "asyncpg-listen==0.0.4",
        "cloudpickle==2.2.0",
        "seaborn==0.11.2",
        "numpy==1.21.0",
        "Pillow==9.1.1",
        "pydantic==1.9.1",
        "catboost==1.1",
        # "/home/ubuntu/ross/pipeline-stack/pipeline",
        "git+https://github.com/mystic-ai/pipeline.git@ross/envs",
        # "dill==0.3.6",
    ],
)

worker_torch_environment = Environment(
    environment_name="worker-torch-environment",
    dependencies=[
        "torch==1.13.0",
        "torchvision==0.14.0",
        "torchaudio==0.13.0",
    ],
    extend_environments=[base_worker_environment],
)
