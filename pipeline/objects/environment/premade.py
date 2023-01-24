from pipeline.objects.environment import Environment

default_worker_environment = Environment(
    name="default-worker-environment",
    dependencies=[
        "torch==1.13.0",
        "torchvision==0.14.0",
        "torchaudio==0.13.0",
        "transformers==4.21.2",
        "opencv-python==<=4.5.5",
        "tensorflow==2.9.1",
        "tensorflow-hub==0.12.0",
        "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@857d5de21a7789d1bba46694cf608b1cb2ea128a",
        "deepspeed==0.5.10"
        "seaborn==0.11.2"
        "numpy==1.21.0",
        "Pillow==9.2.0",
        "spacy['cuda113']==3.4.3",
        "onnxruntime-gpu==1.12.1",
        "sentence-transformers==2.2.2",
        "accelerate==0.10.0",
        "diffusers @ git+https://github.com/huggingface/diffusers.git@5755d16868ec3da7d5eb4f42db77b01fac842ea8"
        "xgboost==1.6.2",
        "einops==0.4.1",
        "wandb==0.13.4",
        "scikit-learn==1.1.2",
        "catboost==1.1",
        "pywhisper==1.0.6",
    ],
)


worker_torch_environment = Environment(
    name="worker-torch-environment",
    dependencies=[
        "torch==1.13.0",
        "torchvision==0.14.0",
        "torchaudio==0.13.0",
    ],
)
