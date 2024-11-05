#!/bin/bash

TRAIN_FLAGS="--batch_size 1 --epochs 50" 
data_dir="/home/vincent18/MCE_dataset/"

python ../test.py --save_dir ../../results/ --model 2DUnet \
    --src_dir ${data_dir} --key "kmeans" \
    $TRAIN_FLAGS --gid "0" --load_model_name "0908-162510"