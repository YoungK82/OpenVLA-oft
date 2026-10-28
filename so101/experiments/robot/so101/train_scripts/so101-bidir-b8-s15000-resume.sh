#!/usr/bin/env bash
# Continues the so101-bidir-b8-s10000 lineage from its 10000-step checkpoint to 15000 steps,
# saving at 12500 and 15000 (same run name, so checkpoints land next to the 2500..10000 ones).
# Launched automatically by chain_to_15000.sh once the 10000-step run has finished.
set -euo pipefail
cd /home/yhkim/openvla-oft
source /home/yhkim/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONPATH=/home/yhkim/openvla-oft
# No --attn_implementation: keep the moojink fork's default bidirectional sdpa, matching
# how this lineage was trained (see training_attention.json in it).
#
# --resume_step 10000 continues log_step/checkpoint numbering, so saves land at 12500 and 15000.
#
# --resume only restores model/action-head/proprio-projector weights, not optimizer or LR
# scheduler state (finetune.py has no such checkpoint). Steps 5000-10000 ran at the decayed
# LR (5e-5), so keep that LR and push num_steps_before_decay past max_steps.
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --robot_platform so101 \
  --vla_path checkpoints/so101-bidir-b8-s10000--10000_chkpt \
  --data_root_dir /home/yhkim/datasets/rlds \
  --dataset_name so101_block_into_cup_50_v5 \
  --run_root_dir checkpoints \
  --run_id_override so101-bidir-b8-s10000 \
  --resume True --resume_step 10000 \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 3 --use_proprio True \
  --batch_size 8 --learning_rate 5e-5 \
  --num_steps_before_decay 100000 --max_steps 15000 \
  --use_val_set True --val_freq 2500 \
  --save_freq 2500 --save_latest_checkpoint_only False \
  --image_aug True --lora_rank 32 --merge_lora_during_training True \
  --wandb_entity local --wandb_project so101-openvla-oft \
  --run_id_note bidir-trimmed-v1.1.0-b8-s15k-resume-from-10000 \
  2>&1 | tee outputs/train-logs/so101-bidir-b8-s15000.log
