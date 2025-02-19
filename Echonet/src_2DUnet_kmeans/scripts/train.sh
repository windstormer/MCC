#!/bin/bash

TRAIN_FLAGS="--batch_size 1 --epochs 50" 
data_dir="/home/vincent18/EchoNet-Dynamic/"

python ../train.py --save_dir ../../results/ --model 2DUnet \
    --src_dir ${data_dir} \
    $TRAIN_FLAGS --gid "1"