from pipeline.cloud.environments import create_environment

env_name = "falcon_7b"
try:
    env_id = create_environment(
        name=env_name,
        python_requirements=[
            "torch",
            "transformers",
            "scipy",
            "accelerate",
            "einops",
            "xformers",
            "bitsandbytes",
        ],
    )
    print(f"New environment ID = {env_id}")
    print(
        "Environment will be pre-emptively cached on compute resources so please "
        "wait a few mins before using..."
    )
except Exception:
    print("Environment already exists, using existing environment...")
