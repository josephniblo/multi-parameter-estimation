#!/bin/bash

# Abort on error
set -e

pipenv run python ./scripts/prepare_and_estimate_states.py
pipenv run papermill ./post-processing/estimate_states.ipynb /dev/null
pipenv run papermill ./post-processing/analyse_estimates.ipynb /dev/null