from pathlib import Path


def _read_template(template_name: str) -> str:
    template_path = Path(__file__).parent / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name} not found")

    return template_path.read_text()


dockerfile_template = _read_template("dockerfile_template.txt")
pipeline_template_python = _read_template("pipeline_template.py")
