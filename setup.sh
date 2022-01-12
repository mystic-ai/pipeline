cat > .git/hooks/pre-push <<- "EOF"
#!/bin/bash
pytest
EOF

poetry install

pre-commit install

pre-commit autoupdate
