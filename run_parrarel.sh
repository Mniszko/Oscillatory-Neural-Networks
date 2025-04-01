#!/bin/bash

NUM_ITERATIONS=250
NUM_EPOCHS=1500
FEATURE_MUL=100
FEATURE_CON=-30
LABEL_MULTI=0.3

echo "Before running, remember to set methods in main files to those corresponding to correct datasets and set appropriate size!"

# Function to run a simulation
run_simulation() {
    local script=$1
    local exp_name=$2
    local num_neurons=$3
    local learning_rate=$4
    local high_value=$5
    local beta=$6
    local num_iterations=$7
    echo "Running simulation for $exp_name"
    for i in $(seq 1 $num_iterations); do
        echo "      Running simulation number $i"
        python $script $exp_name $num_neurons y $NUM_EPOCHS $learning_rate r 0 $FEATURE_MUL $FEATURE_CON $LABEL_MULTI $high_value $beta
    done
}

export -f run_simulation
export NUM_ITERATIONS NUM_EPOCHS FEATURE_MUL FEATURE_CON LABEL_MULTI

# Create a list of commands to run
commands=(
    "run_simulation SL-training.py exp01-10-03-2025 12 0.2 120 0.1 100"
    "run_simulation SL-training.py exp02-10-03-2025 12 0.2 150 0.1 100"
    "run_simulation SL-training.py exp03-10-03-2025 7 0.2 120 0.1 100"
    "run_simulation SL-training.py exp04-10-03-2025 7 0.2 150 0.1 100"
)
# Run commands in parallel using xargs
printf "%s\n" "${commands[@]}" | xargs -P 10 -I {} bash -c "{} || true"

echo "All simulations completed!"
