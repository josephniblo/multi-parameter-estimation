# create a dictionary of strings
declare -A target_matrices

target_matrices=(
    ["D"]="[ [ 0.5,0.5],[0.5,0.5] ]"
    ["A"]="[ [ 0.5,-0.5],[-0.5,0.5] ]"
    ["R"]="[ [ 0.5,-0.5j],[0.5j,0.5] ]"
    ["L"]="[ [ 0.5,0.5j],[-0.5j,0.5] ]"
    ["H"]="[ [ 1,0],[0,0] ]"
    ["V"]="[ [ 0,0],[0,1] ]"
)

# Loop through all the directories in ./data
for dir in ./data/*; do
    # Check if the directory is a directory
    if [ -d "$dir" ]; then
        # check that the directory name contains '1qb'
        if [[ "$dir" == *"1qb"* ]]; then
            # run pipenv run python ./scripts/analyze_single_qubit_tomo.py ./data/<directory> "target_matrix"
            # where target_matrix is the matrix corresponding to the last part of the directory name "..._<target_matrix>"
            # get the last part of the directory name
            last_part=${dir##*/}
            # get the target matrix from the last part of the directory name
            target_matrix=${last_part##*_}
            # check if the target matrix is in the dictionary
            if [[ -v target_matrices[$target_matrix] ]]; then
                # run the python script with the target matrix
                echo "Running analysis for $dir with target matrix $target_matrix"
                echo "Will run: pipenv run python ./scripts/analyze_single_qubit_tomo.py $dir ${target_matrices[$target_matrix]}"
                pipenv run python ./scripts/analyze_single_qubit_tomo.py "$dir" "${target_matrices[$target_matrix]}"
            else
                echo "Target matrix $target_matrix not found in dictionary"
            fi
        fi
    fi
done