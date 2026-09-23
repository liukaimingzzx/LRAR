#!/bin/sh

# n_shot and a_shot settings
# Get command line arguments
gpu=$1
save_dir=$2
fold=$3
data_mode=${4:-mvtec_visa} # mvtec_visa | mvtec_busi | mvtec_brats

# Create the save directory if it doesn't exist
mkdir -p "$save_dir"

# Loop through each combination of n_shot and a_shot
for n in 1 2 4; do
  for a in 1; do
    # Generate a distinct port for torchrun (PID-based, so concurrent runs won't collide)
    port=$((20000 + $$ % 10000 + n))

    echo "Starting training for n_shot=${n}, a_shot=${a}, data_mode=${data_mode} on GPU ${gpu}, saving to ${save_dir}"

    # Run the training command
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$port train.py \
      --data_root /path/to/dataset \
      --data_mode "$data_mode" \
      --fold "$fold" \
      --epoch 20 \
      --batch_size 8 \
      --image_size 448 \
      --print_freq 50 \
      --n_shot "$n" \
      --a_shot "$a" \
      --num_learnable_proxies 25 \
      --save_path "$save_dir" \
      | tee "${save_dir}/n_${n}_a_${a}.log" # Redirect output to a log file
  done
done

echo "All training settings completed."
