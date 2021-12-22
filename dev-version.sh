#!/bin/bash

BRANCH=$(git branch --show-current)
CLEAN_BRANCH=${BRANCH//\/}
BASE_VERSION=$(sed -n '/version/p' pyproject.toml)
SUBSTR=${BASE_VERSION%\"}
VERSION="${SUBSTR}+${CLEAN_BRANCH}\""
sed -i.bak "s|.*version.*|$VERSION|" pyproject.toml
