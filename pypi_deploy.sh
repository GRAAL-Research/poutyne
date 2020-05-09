#!/bin/sh

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ $current_branch != "stable" ]; then
    echo "The current branch must be 'stable'."
    exit 1
fi

POUTYNE_RELEASE_BUILD=1 python setup.py sdist bdist_wheel
python -m twine upload dist/*
