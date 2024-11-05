#!/bin/bash

TRAIN_FLAGS="--img_size 256" 
data_dir="/home/vincent18/MCE_dataset/"

python ../train.py --save_dir ../../results/ \
    --src_dir ${data_dir} --view A4C --task "kmeans" \
    $TRAIN_FLAGS