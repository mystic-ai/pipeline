#!/bin/bash

BRANCH=$(git branch --show-current)
BASE_VERSION=$(sed -n '/version/p' pyproject.toml)
SUBSTR=${BASE_VERSION%\"}
VERSION="${SUBSTR}+${BRANCH}\""
sed -i.bak "s|.*version.*|$VERSION|" pyproject.toml
