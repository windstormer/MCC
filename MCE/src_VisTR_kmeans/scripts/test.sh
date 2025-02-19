#!/bin/bash

TRAIN_FLAGS="--batch_size 1 --epochs 50 --img_size 256" 
data_dir="/home/vincent18/MCE_dataset/"

python ../test.py --save_dir ../../results/SS_10%/ --model VisTR \
    --src_dir ${data_dir} --view "A2C" --key "kmeans" \
    $TRAIN_FLAGS --gid "0" --load_model_name "0108-183431"