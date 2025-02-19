#!/bin/bash

TRAIN_FLAGS="--batch_size 1 --epochs 40" 
data_dir="/home/vincent18/EchoNet-Dynamic/"

python ../train.py --save_dir ../../results/ --model VisTR \
    --src_dir ${data_dir} \
    $TRAIN_FLAGS --gid "0"