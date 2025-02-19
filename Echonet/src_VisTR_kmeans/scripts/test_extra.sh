#!/bin/bash

TRAIN_FLAGS="--batch_size 1 --epochs 50" 
data_dir="/home/vincent18/EchoNet-Dynamic_test/"

python ../test_extra.py --save_dir ../../results/SS/ --model VisTR \
    --src_dir ${data_dir} \
    $TRAIN_FLAGS --gid "1" --load_model_name "0213-181545"