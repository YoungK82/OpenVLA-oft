#!/usr/bin/env bash
# Fine-tune OpenVLA-OFT on the magnet crane RLDS dataset on ONE GPU: 3 exo cameras + task language (FiLM) + 4D state proprio [load, x, y, z].
#
# Usage (run in tmux):
#   CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/datasets/rlds BASE_MODEL=/path/to/openvla-7b bash experiments/robot/crane/train_crane.sh
# Two GPUs: CUDA_VISIBLE_DEVICES=0,1 NPROC=2 BATCH_SIZE=4 GRAD_ACCUM=1 ... (effective batch = BATCH_SIZE x NPROC x GRAD_ACCUM).
# Stop and resume: `touch <RUN_ROOT>/<RUN_ID>/STOP` saves the resumable state and exits; running the same command again
# continues exactly from <RUN_ROOT>/<RUN_ID>--resume_latest (LoRA, heads, FiLM backbone, optimizer, LR schedule, step).
# STOP_AT=HH:MM writes STOP by itself at that clock time (e.g. STOP_AT=09:50). Extra finetune.py options can follow.
#
# Schedule = the Hawk recipe (slurm_train_crane.sbatch) so runs compare: effective batch 8 = BATCH_SIZE 4 x GRAD_ACCUM 2
# (batch 8 OOMs on a 96GB RTX PRO 6000; ~18.8K steps per epoch), 25K steps (~1.33 epochs), constant LR 5e-4 (LR_DECAY=1:
# one 10x decay when train L1 plateaus, not before MIN_DECAY_STEP, at the latest DECAY_STEP). Validation + checkpoint
# every SAVE_FREQ; pick the checkpoint by validation (overfitting guard), not by train L1.
#
# b4-2 additions (defaults keep the old behavior):
#   LEARNING_RATE (default 5e-4). Gradient norms (DDP-averaged params / rank 0's action head) are printed on every TRAIN
#   line and logged to W&B; there is no clipping.
#   VAL_DECAY=1: one 10x LR decay when the validation balanced accuracy has not beaten its best by > VAL_DECAY_MIN_DELTA
#   for VAL_DECAY_PATIENCE validations in a row, counted from VAL_DECAY_MIN_STEP on (use instead of LR_DECAY=1).
#   VAL_FREQ (default SAVE_FREQ), VAL_BATCH_SIZE (default 4). Train metrics are now means over all GPUs.
#   A run's settings are fixed at its first start: <RUN_ROOT>/<RUN_ID>/plan.env records LEARNING_RATE,
#   LR_DECAY/DECAY_STEP/MIN_DECAY_STEP, VAL_DECAY*, SAVE_FREQ, VAL_FREQ, VAL_BATCH_SIZE, MAX_STEPS and BATCH_SIZE/GRAD_ACCUM/
#   NPROC (finetune.py also refuses a resume whose effective batch differs from the saved one), and every later start
#   of the same RUN_ID (a resume) reads them back for each variable not set in the environment. So relaunching with the
#   plain command keeps the run's settings; set a variable explicitly to override it (e.g. MAX_STEPS to train longer).
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
DATASET_NAME="${DATASET_NAME:-crane_b_magnet_onoff_state_tvt}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the RLDS root that holds $DATASET_NAME/}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to the openvla-7b directory}"
RUN_ROOT="${RUN_ROOT:-$repo_dir/checkpoints}"
RUN_ID="${RUN_ID:-$DATASET_NAME}"
plan_file="$RUN_ROOT/$RUN_ID/plan.env"
plan_keys=(LEARNING_RATE LR_DECAY DECAY_STEP MIN_DECAY_STEP VAL_DECAY VAL_DECAY_PATIENCE VAL_DECAY_MIN_DELTA
           VAL_DECAY_MIN_STEP SAVE_FREQ VAL_FREQ VAL_BATCH_SIZE MAX_STEPS BATCH_SIZE GRAD_ACCUM NPROC)
if [[ -f "$plan_file" ]]; then
  while IFS='=' read -r key value; do
    [[ " ${plan_keys[*]} " == *" $key "* ]] || continue
    [[ -n "${!key+x}" ]] || export "$key=$value"
  done < "$plan_file"
  echo "Run settings from $plan_file (environment variables override): $(tr '\n' ' ' < "$plan_file")"
fi
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NPROC="${NPROC:-1}"  # GPUs (one process each)
MAX_STEPS="${MAX_STEPS:-25000}"
LR_DECAY="${LR_DECAY:-0}"  # 0 = constant LR; 1 = one automatic 10x decay
DECAY_STEP="${DECAY_STEP:-20000}"
MIN_DECAY_STEP="${MIN_DECAY_STEP:-10000}"
[[ "$LR_DECAY" == "1" ]] || DECAY_STEP=1000000000  # milestone never reached
SAVE_FREQ="${SAVE_FREQ:-5000}"
VAL_FREQ="${VAL_FREQ:-$SAVE_FREQ}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
VAL_DECAY="${VAL_DECAY:-0}"
VAL_DECAY_PATIENCE="${VAL_DECAY_PATIENCE:-3}"
VAL_DECAY_MIN_DELTA="${VAL_DECAY_MIN_DELTA:-0.005}"
VAL_DECAY_MIN_STEP="${VAL_DECAY_MIN_STEP:-5000}"
if [[ "$LR_DECAY" == "1" && "$VAL_DECAY" == "1" ]]; then
  echo "Set only one of LR_DECAY=1 (train L1 plateau) and VAL_DECAY=1 (validation plateau)." >&2
  exit 2
fi
RESUME_SAVE_FREQ="${RESUME_SAVE_FREQ:-250}"
STOP_AT="${STOP_AT:-}"

run_dir="$RUN_ROOT/$RUN_ID"
resume_dir="$run_dir--resume_latest"
mkdir -p "$run_dir"
if [[ ! -f "$plan_file" ]]; then
  for key in "${plan_keys[@]}"; do printf '%s=%s\n' "$key" "${!key}"; done > "$plan_file"
  echo "Wrote run settings to $plan_file"
fi
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
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" vla-scripts/finetune.py \
  --robot_platform crane --vla_path "$BASE_MODEL" \
  --data_root_dir "$DATA_ROOT" --dataset_name "$DATASET_NAME" --run_root_dir "$RUN_ROOT" --run_id_override "$RUN_ID" \
  --use_l1_regression True --use_diffusion False \
  --use_film True --use_proprio True --num_images_in_input 3 \
  --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate "$LEARNING_RATE" --lr_warmup_steps 0 \
  --auto_lr_decay "$( [[ "$LR_DECAY" == "1" ]] && echo True || echo False )" --min_steps_before_auto_decay "$MIN_DECAY_STEP" \
  --num_steps_before_decay "$DECAY_STEP" --max_steps "$MAX_STEPS" \
  --use_val_set True --val_freq "$VAL_FREQ" --val_batch_size "$VAL_BATCH_SIZE" \
  --val_plateau_decay "$( [[ "$VAL_DECAY" == "1" ]] && echo True || echo False )" --val_plateau_patience "$VAL_DECAY_PATIENCE" \
  --val_plateau_min_delta "$VAL_DECAY_MIN_DELTA" --val_plateau_min_step "$VAL_DECAY_MIN_STEP" \
  --save_freq "$SAVE_FREQ" --save_latest_checkpoint_only False \
  --resume_save_freq "$RESUME_SAVE_FREQ" ${resume_args[@]+"${resume_args[@]}"} \
  --merge_lora_during_training False --image_aug True --lora_rank 32 \
  --wandb_entity "${WANDB_ENTITY:-local}" --wandb_project crane-openvla-oft "$@" \
  2>&1 | tee -a "$run_dir/train.log"
