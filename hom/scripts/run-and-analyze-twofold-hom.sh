#!/bin/bash

# This script automates the process of running a twofold HOM and analyzing the results.
# Run it from ./hom directory.

# Usage:
#   ./run-and-analyze-twofold-hom.sh <Temperature> <Power> <Subfolder>
# Arguments:
#   <Temperature> - The temperature parameter for the simulation.
#   <Power>       - The power parameter for the simulation.
#   <Subfolder>   - The subfolder within the ./data directory where the output file will be moved.

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <Temperature> <Power>"
    exit 1
fi

TEMPERATURE="$1"
POWER="$2"
SUBFOLDER="$3"

# Execute the Python script
pipenv run python ./scripts/run-twofold-hom.py "$TEMPERATURE" "$POWER"

# Get the filename of the output file,
# which is the last file in ./data
OUTPUT_FILE=$(ls -t ./data | head -n 1)

pipenv run python ./post-processing/fit-hom.py "./data/$OUTPUT_FILE"

# move the output file to the results directory
mv "./data/$OUTPUT_FILE" "./data/$SUBFOLDER/$OUTPUT_FILE"