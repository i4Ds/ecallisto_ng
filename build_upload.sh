#!/bin/bash

echo "Building the package..."
python -m build

echo "Uploading the package to PyPi..."
python -m twine upload --repository pypi dist/*
