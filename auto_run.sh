#!/bin/bash

NUM_ITERATIONS=20
NUM_EPOCHS=300
NUM_NEURONS = 6

K_LEARNING_RATE = 1

SL_LEARNING_RATE = 0.2

echo before running remember to set methods in main files to those corresponding to correct datasets and set appropriate size!

for i in $(seq 1 $NUM_ITERATIONS); do
    echo "Running number $i"
    python SL-training.py SL-output $NUM_NEURONS y $NUM_EPOCHS $SL_LEARNING_RATE
done

for i in $(seq 1 $NUM_ITERATIONS); do
    echo "Running number $i"
        python3 K-training.py $NUM_NEURONS y $NUM_EPOCHS $K_LEARNING_RATE
done

