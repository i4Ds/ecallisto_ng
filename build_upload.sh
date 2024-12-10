#!/bin/bash

echo "Removing old builds..."
rm -rf dist/
rm -rf build/
echo "Building the package..."
python -m build

echo "Uploading the package to PyPi..."
python -m twine upload --repository ecallisto_ng dist/* --verbose
