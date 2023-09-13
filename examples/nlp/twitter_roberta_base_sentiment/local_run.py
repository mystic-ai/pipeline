from .upload import pipeline_graph

output = pipeline_graph.run(
    "@plutopulp Mystic is #awesome 👍 https://github.com/mystic-ai"
)

print(output)
