#!/bin/bash

# delete artefacts created during testing
# Must be run inside container or with sudo
rm -rf .hypothesis
rm -rf .pytest_cache
find . -type d -name "__pycache__" -exec rm -rf {} +
