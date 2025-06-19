#!/bin/bash

# Abort on error
set -e

pipenv run python ./scripts/prepare_and_estimate_states_with_tomtag.py

pipenv run papermill ./post-processing/1.0.0_sort_tags.ipynb /dev/null
pipenv run papermill ./post-processing/1.0.1_label_coincidences.ipynb /dev/null
pipenv run papermill ./post-processing/1.1_sum_scale_and_chunk.ipynb /dev/null
pipenv run papermill ./post-processing/4.1_estimate_on_corrected_chunks.ipynb /dev/null
pipenv run papermill ./post-processing/5.1_analyse_corrected_estimates.ipynb /dev/null