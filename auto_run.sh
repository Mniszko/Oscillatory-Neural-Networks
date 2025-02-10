#!/bin/bash

NUM_ITERATIONS=2
NUM_EPOCHS=10
NUM_NEURONS=7

K_LEARNING_RATE=1

SL_LEARNING_RATE=0.4

echo before running remember to set methods in main files to those corresponding to correct datasets and set appropriate size!

for i in $(seq 1 $NUM_ITERATIONS); do
    echo "      Running simulation number $i"
    python3 SL-training.py SL-output $NUM_NEURONS y $NUM_EPOCHS $SL_LEARNING_RATE
done

for i in $(seq 1 $NUM_ITERATIONS); do
    echo "      Running simulation number $i"
    python3 K-training.py K-output $NUM_NEURONS y $NUM_EPOCHS $K_LEARNING_RATE
done

