#! /bin/bash

cd $(dirname $0)

rm -rf ./outer_map_input

mkdir outer_map_input
mkdir outer_map_input/detection_result
mkdir outer_map_input/ground_truth

sudo python3 ./predict_voc_with_eval.py
sudo python3 ./compute_map.py

rm -rf ./outer_map_input

# nohup ./run.sh &