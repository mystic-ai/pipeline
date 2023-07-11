from pipeline.v3.pipelines import run_pipeline

output = run_pipeline(
    "ph/module_test:main",
    1,
)

print(output)
