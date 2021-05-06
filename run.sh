#! /bin/bash

cd $(dirname $0)

mkdir ./logs
sudo python3 train.py

# nohup sudo ./run.sh &