#!/usr/bin/env bash
# Fine-tune OpenVLA-OFT on the magnet crane RLDS dataset on ONE GPU: 3 exo cameras + language (FiLM) + 4D state proprio.
#
# Usage (run in tmux):
#   CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/datasets/rlds BASE_MODEL=/path/to/openvla-7b bash experiments/robot/crane/train_crane.sh
# Stop and resume: `touch <RUN_ROOT>/<RUN_ID>/STOP` saves the resumable state and exits; running the same command again
# continues exactly from <RUN_ROOT>/<RUN_ID>--resume_latest (LoRA, heads, FiLM backbone, optimizer, LR schedule, step).
# STOP_AT=HH:MM writes STOP by itself at that clock time (e.g. STOP_AT=09:50). Extra finetune.py options can follow.
#
# Schedule (OFT+ recipe on 1 GPU): batch 8 (no accumulation; ~18.8K steps per epoch), LR 5e-4, one 10x decay when train L1
# plateaus (not before MIN_DECAY_STEP, at the latest DECAY_STEP), MAX_STEPS cap. Validation + checkpoint every SAVE_FREQ;
# pick the checkpoint by validation (overfitting guard), not by train L1.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
DATASET_NAME="${DATASET_NAME:-crane_b_magnet_onoff_nopower}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the RLDS root that holds $DATASET_NAME/}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to the openvla-7b directory}"
RUN_ROOT="${RUN_ROOT:-$repo_dir/checkpoints}"
RUN_ID="${RUN_ID:-$DATASET_NAME}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-120000}"
DECAY_STEP="${DECAY_STEP:-80000}"
MIN_DECAY_STEP="${MIN_DECAY_STEP:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
RESUME_SAVE_FREQ="${RESUME_SAVE_FREQ:-250}"
STOP_AT="${STOP_AT:-}"

run_dir="$RUN_ROOT/$RUN_ID"
resume_dir="$run_dir--resume_latest"
mkdir -p "$run_dir"
rm -f "$run_dir/STOP"  # a STOP left by the previous job would end this one at once
resume_args=()
if [[ -f "$resume_dir/training_state.pt" ]]; then
  resume_args=(--resume_from "$resume_dir")
  echo "Resuming from $resume_dir"
fi
if [[ -n "$STOP_AT" ]]; then
  stop_epoch="$(date -d "$STOP_AT" +%s)"
  (( stop_epoch <= $(date +%s) )) && stop_epoch="$(date -d "tomorrow $STOP_AT" +%s)"
  ( sleep $(( stop_epoch - $(date +%s) )) && touch "$run_dir/STOP" ) >/dev/null 2>&1 &
  timer_pid=$!
  trap 'kill "$timer_pid" 2>/dev/null || true' EXIT
  echo "Will save and stop at $(date -d "@$stop_epoch" '+%F %T')"
fi

cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE="${WANDB_MODE:-offline}" WANDB_DIR="${WANDB_DIR:-$RUN_ROOT}"
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --robot_platform crane --vla_path "$BASE_MODEL" \
  --data_root_dir "$DATA_ROOT" --dataset_name "$DATASET_NAME" --run_root_dir "$RUN_ROOT" --run_id_override "$RUN_ID" \
  --use_l1_regression True --use_diffusion False \
  --use_film True --use_proprio True --num_images_in_input 3 \
  --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 --lr_warmup_steps 0 \
  --auto_lr_decay True --min_steps_before_auto_decay "$MIN_DECAY_STEP" \
  --num_steps_before_decay "$DECAY_STEP" --max_steps "$MAX_STEPS" \
  --use_val_set True --val_freq "$SAVE_FREQ" --save_freq "$SAVE_FREQ" --save_latest_checkpoint_only False \
  --resume_save_freq "$RESUME_SAVE_FREQ" ${resume_args[@]+"${resume_args[@]}"} \
  --merge_lora_during_training False --image_aug True --lora_rank 32 \
  --wandb_entity "${WANDB_ENTITY:-local}" --wandb_project crane-openvla-oft "$@" \
  2>&1 | tee -a "$run_dir/train.log"
