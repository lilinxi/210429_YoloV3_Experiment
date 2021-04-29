#! /bin/bash

cd $(dirname $0)

ps -aux | grep "train" | awk '{ print $2 }' | sudo xargs kill -9

# ps -aux | grep "train.py" | awk '{ print $2 }' | sudo xargs kill -9
# ps -aux | grep "train_cpy.py" | awk '{ print $2 }' | sudo xargs kill -9