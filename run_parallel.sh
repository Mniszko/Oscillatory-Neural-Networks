#!/bin/bash

NUM_ITERATIONS=50
NUM_EPOCHS=1000
FEATURE_MUL=100
FEATURE_CON=-30
LABEL_MULTI=0.3
LEARNING_RATE=0.15
BETA_VALUE=0.1
NUM_NEURONS=7

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
    "run_simulation SL-training.py exp01-08-05-2025 $NUM_NEURONS $LEARNING_RATE 0.1 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp02-08-05-2025 $NUM_NEURONS $LEARNING_RATE 0.5 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp03-08-05-2025 $NUM_NEURONS $LEARNING_RATE 1 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp04-08-05-2025 $NUM_NEURONS $LEARNING_RATE 2 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp05-08-05-2025 $NUM_NEURONS $LEARNING_RATE 5 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp06-08-05-2025 $NUM_NEURONS $LEARNING_RATE 10 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp07-08-05-2025 $NUM_NEURONS $LEARNING_RATE 30 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp08-08-05-2025 $NUM_NEURONS $LEARNING_RATE 60 $BETA_VALUE $NUM_ITERATIONS"
    "run_simulation SL-training.py exp09-08-05-2025 $NUM_NEURONS $LEARNING_RATE 90 $BETA_VALUE $NUM_ITERATIONS"
)
# Run commands in parallel using xargs
printf "%s\n" "${commands[@]}" | xargs -P 10 -I {} bash -c "{} || true"

echo "All simulations completed!"
