#!/bin/bash

echo "Removing old builds..."
rm -rf dist/
rm -rf build/
echo "Building the package..."
uv run python -m build

echo "Uploading the package to PyPi..."
uv run python -m twine upload --repository ecallisto_ng dist/* --verbose
