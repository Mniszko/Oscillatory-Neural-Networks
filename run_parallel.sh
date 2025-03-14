#!/bin/bash

NUM_ITERATIONS=40
NUM_EPOCHS=1500
FEATURE_MUL=100
FEATURE_CON=-30
LABEL_MULTI=0.3

echo "Before running, remember to set methods in main files to those corresponding to correct datasets and set appropriate size!"

run_simulation_first() {
    local script=$1
    local exp_name=$2
    local num_neurons=$3
    local learning_rate=$4
    echo "Running simulation for $exp_name"
    for i in $(seq 1 $NUM_ITERATIONS); do
        echo "      Running simulation number $i"
        python $script $exp_name $num_neurons y $NUM_EPOCHS $learning_rate r 1 $FEATURE_MUL $FEATURE_CON $LABEL_MULTI
    done
}

export -f run_simulation_first
export NUM_ITERATIONS NUM_EPOCHS FEATURE_MUL FEATURE_CON LABEL_MULTI

# Create a list of commands to run
commands=(
    "run_simulation_first SL-training.py exp01-10-03-2025 12 0.2"
)
# Run commands in parallel using xargs
printf "%s\n" "${commands[@]}" | xargs -P 7 -I {} bash -c "{} || true"

echo "All simulations completed!"

