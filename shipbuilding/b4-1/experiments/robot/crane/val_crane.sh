#!/usr/bin/env bash
# Re-score saved magnet crane checkpoints with the deterministic validation of finetune.py (no training).
#
# Usage (run in tmux, from anywhere):
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 DATA_ROOT=~/datasets/rlds BASE_MODEL=/path/to/openvla-7b \
#     bash experiments/robot/crane/val_crane.sh CKPT_DIR [CKPT_DIR ...]
# CKPT_DIR = a `<run>--<step>_chkpt` (or `--resume_latest`) folder written by train_crane.sh; it must hold
# training_state.pt. BASE_MODEL must be the same unmerged base model the run was trained on (checked).
#
# Every N-th frame (VAL_STRIDE, default 3) of every validation episode is scored once, in a fixed order, split across
# the GPUs and summed, so numbers from different checkpoints compare directly. Prints VAL lines (loss, L1, on/off
# accuracy, per-magnet ON/OFF recall) and writes CKPT_DIR/val_metrics_stride<N>.json. The training run folder,
# its train.log and --resume_latest are not touched (outputs go to <RUN_ROOT>/<DATASET_NAME>--val/).
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 CHECKPOINT_DIRECTORY [CHECKPOINT_DIRECTORY ...]" >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
DATASET_NAME="${DATASET_NAME:-crane_b_magnet_onoff_state_tvt}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the RLDS root that holds $DATASET_NAME/}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to the openvla-7b directory the checkpoints were trained on}"
RUN_ROOT="${RUN_ROOT:-$repo_dir/checkpoints}"
NPROC="${NPROC:-1}"
VAL_STRIDE="${VAL_STRIDE:-3}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"

checkpoints=()
for checkpoint in "$@"; do
  checkpoint="$(cd -- "$checkpoint" && pwd -P)"
  [[ -f "$checkpoint/training_state.pt" ]] || { echo "$checkpoint has no training_state.pt" >&2; exit 2; }
  checkpoints+=("$checkpoint")
done

val_id="$DATASET_NAME--val"
log="$RUN_ROOT/$val_id/val.log"
mkdir -p "$RUN_ROOT/$val_id"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled
for checkpoint in "${checkpoints[@]}"; do
  echo "[$(date '+%F %T')] Validating $checkpoint (stride $VAL_STRIDE, $NPROC GPU(s), batch $VAL_BATCH_SIZE)" | tee -a "$log"
  torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" vla-scripts/finetune.py \
    --robot_platform crane --vla_path "$BASE_MODEL" \
    --data_root_dir "$DATA_ROOT" --dataset_name "$DATASET_NAME" --run_root_dir "$RUN_ROOT" --run_id_override "$val_id" \
    --use_l1_regression True --use_diffusion False \
    --use_film True --use_proprio True --num_images_in_input 3 \
    --batch_size "$VAL_BATCH_SIZE" --lr_warmup_steps 0 --lora_rank 32 --merge_lora_during_training False \
    --use_val_set True --val_only True --resume_from "$checkpoint" \
    --val_frame_stride "$VAL_STRIDE" --val_batch_size "$VAL_BATCH_SIZE" \
    --wandb_entity local --wandb_project crane-openvla-oft \
    2>&1 | tee -a "$log"
done
echo "[$(date '+%F %T')] Done. Results: CKPT_DIR/val_metrics_stride$VAL_STRIDE.json; log $log" | tee -a "$log"
