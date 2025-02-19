#!/bin/bash

TRAIN_FLAGS="--batch_size 1 --epochs 50 --img_size 256" 
data_dir="/home/vincent18/MCE_dataset/"

python ../train.py --save_dir ../../results/ --model 2DUnet \
    --src_dir ${data_dir} --key "kmeans" \
    $TRAIN_FLAGS --gid "1"