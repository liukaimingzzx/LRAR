#!/bin/bash

mode=$1  # "train" or "test"
gpu=$2
save_dir=$3
fold=$4
test_dataset=$5  # "mvtec" | "visa" | "busi" | "brats"
data_mode=${6:-mvtec_visa}  # "mvtec_visa" | "mvtec_busi" | "mvtec_brats"

if [ "$mode" = "train" ]; then
    sh scripts/train.sh $gpu $save_dir $fold $data_mode
fi

sh scripts/test.sh $gpu $save_dir $test_dataset
