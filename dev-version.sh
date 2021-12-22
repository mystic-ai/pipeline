#!/bin/bash

BRANCH=$(git branch --show-current)
HASH=$(md5 <<<${BRANCH})
BASE_VERSION=$(sed -n '/version/p' pyproject.toml)
SUBSTR=${BASE_VERSION%\"}
VERSION="${SUBSTR}+${HASH}\""
sed -i.bak "s|.*version.*|$VERSION|" pyproject.toml
