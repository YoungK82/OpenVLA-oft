#!/usr/bin/env bash
set -euo pipefail
cd /home/yhkim/openvla-oft
source /home/yhkim/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONPATH=/home/yhkim/openvla-oft
# No --attn_implementation: the moojink fork's default sdpa is bidirectional (official OpenVLA-OFT).
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --robot_platform so101 \
  --vla_path /home/shared/openvla-oft/models/openvla-7b-47a0ec7 \
  --data_root_dir /home/yhkim/datasets/rlds \
  --dataset_name so101_block_into_cup_50_v5 \
  --run_root_dir checkpoints \
  --run_id_override so101-bidir-b8-s10000 \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 3 --use_proprio True \
  --batch_size 8 --learning_rate 5e-4 \
  --num_steps_before_decay 5000 --max_steps 10000 \
  --use_val_set True --val_freq 2500 \
  --save_freq 2500 --save_latest_checkpoint_only False \
  --image_aug True --lora_rank 32 --merge_lora_during_training True \
  --wandb_entity local --wandb_project so101-openvla-oft \
  --run_id_note bidir-trimmed-v1.1.0-b8-s10k \
  2>&1 | tee outputs/train-logs/so101-bidir-b8-s10000.log
