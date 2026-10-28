#!/usr/bin/env bash
# Resumes so101-bidir-b8-s10000 from the 5000-step checkpoint (the 10000-step run was
# terminated at ~5519 steps to free the GPU; only the 2500/5000 checkpoints were saved).
# Run later with: bash outputs/train-logs/so101-bidir-b8-s10000-resume.sh
set -euo pipefail
cd /home/yhkim/openvla-oft
source /home/yhkim/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONPATH=/home/yhkim/openvla-oft
# No --attn_implementation: keep the moojink fork's default bidirectional sdpa, matching
# how so101-bidir-b8-s10000--5000_chkpt was trained (see training_attention.json in it).
#
# --resume_step 5000 continues log_step/checkpoint numbering from the 5000 checkpoint, so the
# next saves land at 7500 and 10000 under the SAME run_id_override as the original run.
#
# --resume only restores model/action-head/proprio-projector weights, not optimizer or LR
# scheduler state (finetune.py has no such checkpoint). The original run's decay milestone
# (step 5000) was already reached, so we resume directly at the decayed LR (5e-5) and push
# num_steps_before_decay past max_steps so it never fires again.
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --robot_platform so101 \
  --vla_path checkpoints/so101-bidir-b8-s10000--5000_chkpt \
  --data_root_dir /home/yhkim/datasets/rlds \
  --dataset_name so101_block_into_cup_50_v5 \
  --run_root_dir checkpoints \
  --run_id_override so101-bidir-b8-s10000 \
  --resume True --resume_step 5000 \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 3 --use_proprio True \
  --batch_size 8 --learning_rate 5e-5 \
  --num_steps_before_decay 100000 --max_steps 10000 \
  --use_val_set True --val_freq 2500 \
  --save_freq 2500 --save_latest_checkpoint_only False \
  --image_aug True --lora_rank 32 --merge_lora_during_training True \
  --wandb_entity local --wandb_project so101-openvla-oft \
  --run_id_note bidir-trimmed-v1.1.0-b8-s10k-resume-from-5000 \
  2>&1 | tee -a outputs/train-logs/so101-bidir-b8-s10000.log
