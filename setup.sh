G='\033[0;32m'
Y='\033[1;33m'
NC='\033[0m'

echo -e "${Y}Setting Pre-Push hooks...${NC}"
cat > .git/hooks/pre-push <<- "EOF"
#!/bin/bash
pytest
EOF

chmod +x .git/hooks/pre-push
echo -e "${G}Pre-Push hooks set.${NC}"

echo -e "${Y}Installing deps...${NC}"
poetry install
echo -e "${G}Dependencies installed.${NC}"
echo -e "${Y}Setting Pre-Commit hooks...${NC}"
pre-commit install

pre-commit autoupdate
echo -e "${G}Pre-Commit hooks set.${NC}"
