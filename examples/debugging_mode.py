from pipeline.cloud.pipelines import run_pipeline
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(True)

output = run_pipeline("test:v5", 5)
