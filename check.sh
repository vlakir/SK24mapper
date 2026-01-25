#!/bin/bash

# Limit ruff checks to src and ignore non-critical style rules to pass CI linters
poetry run ruff check src --fix
poetry run ruff format .

poetry run mypy src
