#!/bin/bash

TRAIN_FLAGS="--batch_size 1 --epochs 40" 
data_dir="/home/vincent18/EchoNet-Dynamic/"

python ../test.py --save_dir ../../results/SS/ --model VisTR \
    --src_dir ${data_dir} \
    $TRAIN_FLAGS --gid "0" --load_model_name "0213-181545"