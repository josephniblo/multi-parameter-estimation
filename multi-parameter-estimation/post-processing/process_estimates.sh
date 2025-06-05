#!/bin/bash

# Abort on error
set -e

pipenv run papermill ./post-processing/1.0.0_sort_tags.ipynb /dev/null
pipenv run papermill ./post-processing/1.0.1_label_coincidences.ipynb /dev/null
pipenv run papermill ./post-processing/1.1_sum_scale_and_chunk.ipynb /dev/null
pipenv run papermill ./post-processing/4_estimate_on_chunks.ipynb /dev/null
pipenv run papermill ./post-processing/5_analyse_estimates.ipynb /dev/null