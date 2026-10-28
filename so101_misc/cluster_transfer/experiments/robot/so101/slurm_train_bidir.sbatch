#!/usr/bin/env bash
#SBATCH --job-name=so101-bidir
#SBATCH --partition=a100_40g
#SBATCH --account=n-gpu
#SBATCH --qos=n-qos
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Hawk rules (n-qos): max 12 h per job, max 8 GPUs at once, 16 cores per GPU (-n = 16 x GPUs), 2/4/8 GPUs preferred.
# Starts on 1 GPU. To change the GPU count from the next job on: `echo 2 > <RUN_ROOT>/<RUN_ID>/NEXT_GPUS`
# (1, 2, 4 or 8); or now: touch STOP, then `sbatch --gres=gpu:2 -n 32 ...`.
#
# SO-101 OpenVLA-OFT fine-tuning with bidirectional attention (the paper's parallel decoding) on A100 40GB GPUs.
#
# - Effective batch 32 (the paper's ALOHA batch) = per-GPU batch 2 x GPUs x gradient accumulation, where the
#   accumulation is derived from the allocated GPU count. Measured per-GPU peaks (3 images + proprio): batch 1 =
#   27 GB, batch 2 = 35 GB, batch 4 = 47 GiB. If batch 2 runs out of memory, use BATCH_SIZE=1.
# - Checkpoints are saved unmerged by default (LoRA adapter + heads + optimizer, ~2.5 GB instead of ~15 GB).
#   Copy one back and merge it on the inference machine (vla-scripts/merge_lora_weights_and_save.py), as the
#   OFT authors recommend merging on the device used for testing.
# - Schedule follows the OFT authors' rule instead of fixed steps: LR 5e-4 until train L1 decreases very slowly
#   (<2% over 2,000 steps, not before step 10,000), then one 10x decay, then train until L1 plateaus below 0.01
#   (the paper's criterion), which saves a final checkpoint and writes <run_dir>/DONE. MAX_STEPS (50,000, the
#   paper's scoop task) caps the run; DECAY_STEP is only the latest allowed decay. Checkpoints every 2,500 steps.
#   AUTO_SCHEDULE=0 uses the fixed MAX_STEPS/DECAY_STEP instead. The plan is stored in <run_dir>/plan.env (editable;
#   later jobs read it) and every later job reuses it.
# - W&B logs offline by default into <RUN_ROOT>/wandb, as one run id (stored in plan.env) that every resubmitted
#   job continues; upload later from the login node with `wandb sync <RUN_ROOT>/wandb/offline-run-*`.
#   WANDB_MODE=online needs `wandb login` and internet on the compute nodes.
# - Resubmitting the same RUN_ID continues exactly from <run_dir>--resume_latest (unmerged LoRA + heads + optimizer).
#   To stop early on purpose: `touch <RUN_ROOT>/<RUN_ID>/STOP`; it saves the resumable state and exits.
# - Per-job time limit: JOB_MINUTES before the limit the job writes STOP itself, saves, and resubmits this script
#   (same partition/account) until MAX_STEPS is reached. A crash or a manual STOP does not resubmit.
set -euo pipefail

# ---- Site paths (override with `sbatch --export=ALL,VAR=...`) ----
WORK_DIR="${WORK_DIR:-$HOME/yhkim/so101-openvla-oft}"   # Hawk layout: openvla-oft/, env/, datasets/, models/
REPO_DIR="${REPO_DIR:-$WORK_DIR/openvla-oft}"
CONDA_SH="${CONDA_SH:-${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}}"
CONDA_ENV="${CONDA_ENV:-$WORK_DIR/env}"
BASE_MODEL="${BASE_MODEL:-$WORK_DIR/models/openvla-7b-47a0ec7}"
DATA_ROOT="${DATA_ROOT:-$WORK_DIR/datasets/rlds}"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/checkpoints}"

# ---- Recipe ----
RUN_ID="${RUN_ID:-so101-bidir-b32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
TRAIN_HOURS="${TRAIN_HOURS:-84}"             # only used when MAX_STEPS is empty
DECAY_FRACTION="${DECAY_FRACTION:-0.8}"      # latest allowed decay = 80% of MAX_STEPS
SAVE_FREQ="${SAVE_FREQ:-2500}"               # checkpoints to copy back and evaluate
AUTO_SCHEDULE="${AUTO_SCHEDULE:-1}"          # 1 = plateau-based decay + stop at L1 < STOP_L1 (paper)
STOP_L1="${STOP_L1:-0.01}"
# EXTRA_ARGS: extra finetune.py options (word-split), e.g. tiny --loss_block_steps for a smoke test of the schedule
RESUME_SAVE_FREQ="${RESUME_SAVE_FREQ:-250}"  # rolling resumable state (lose at most this many steps)
MERGE_ON_CLUSTER="${MERGE_ON_CLUSTER:-False}"
MAX_STEPS="${MAX_STEPS-50000}"               # cap; '' = size by TRAIN_HOURS from a 40-step benchmark
WANDB_ENTITY="${WANDB_ENTITY:-}"             # empty = the logged-in account's default entity
WANDB_PROJECT="${WANDB_PROJECT:-so101-openvla-oft}"
JOB_MINUTES="${JOB_MINUTES:-720}"            # must match #SBATCH --time
STOP_MARGIN_MINUTES="${STOP_MARGIN_MINUTES:-25}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_DIR/experiments/robot/so101/slurm_train_bidir.sbatch}"

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "conda.sh not found; submit from a shell where conda works, or pass CONDA_SH=<conda base>/etc/profile.d/conda.sh" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_DIR="${WANDB_DIR:-$RUN_ROOT}" TF_FORCE_GPU_ALLOW_GROWTH=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN_DIR="$RUN_ROOT/$RUN_ID"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
if (( EFFECTIVE_BATCH % (BATCH_SIZE * NUM_GPUS) != 0 )); then
  echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH is not divisible by BATCH_SIZE=$BATCH_SIZE x NUM_GPUS=$NUM_GPUS" >&2
  exit 1
fi
GRAD_ACCUM=$(( EFFECTIVE_BATCH / (BATCH_SIZE * NUM_GPUS) ))
echo "GPUs=$NUM_GPUS per-GPU batch=$BATCH_SIZE accumulation=$GRAD_ACCUM effective batch=$EFFECTIVE_BATCH"
RESUME_DIR="$RUN_DIR--resume_latest"
mkdir -p "$RUN_DIR"

# Stock transformers makes sdpa causal; that is how the earlier 25k checkpoint ended up causal. Refuse to train.
python - <<'PY'
import importlib.metadata, json
dist = importlib.metadata.distribution("transformers")
direct = dist.read_text("direct_url.json")
source = json.loads(direct).get("url", "") if direct else "pypi"
if "transformers-openvla-oft" not in source:
    raise SystemExit(f"transformers {dist.version} from {source!r} is not the OFT fork; sdpa would be causal.")
print(f"transformers {dist.version} from {source} (bidirectional sdpa)")
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$WANDB_MODE" == "online" ]]; then
  python -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 'W&B is not logged in: run wandb login once, or set WANDB_MODE=offline.')"
fi

common_args=(
  --robot_platform so101 --vla_path "$BASE_MODEL"
  --data_root_dir "$DATA_ROOT" --dataset_name so101_block_into_cup_50_v5
  --run_root_dir "$RUN_ROOT"
  --use_l1_regression True --use_diffusion False --use_film False
  --num_images_in_input 3 --use_proprio True
  --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate "$LEARNING_RATE"
  --image_aug True --lora_rank 32
  --wandb_project "$WANDB_PROJECT"
)
[[ -n "$WANDB_ENTITY" ]] && common_args+=(--wandb_entity "$WANDB_ENTITY")

if [[ -f "$RUN_DIR/plan.env" ]]; then
  source "$RUN_DIR/plan.env"
  echo "Using stored plan: MAX_STEPS=$MAX_STEPS DECAY_STEP=$DECAY_STEP"
else
  if [[ -z "$MAX_STEPS" ]]; then
    WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/finetune.py "${common_args[@]}" \
      --run_id_override "$RUN_ID-benchmark" --max_steps 40 --save_freq 1000000 --use_val_set False \
      --profile_training True --profile_warmup_steps 5 2>&1 | tee "$RUN_DIR/benchmark.log"
    step_seconds="$(grep -oP 'mean_step_seconds=\K[0-9.]+' "$RUN_DIR/benchmark.log")"
    # ~8% headroom for validation, checkpoint merges and data-loader stalls
    MAX_STEPS="$(python -c "s=$step_seconds; f=$SAVE_FREQ; n=int($TRAIN_HOURS*3600*0.92/s); print(max(f, n//f*f))")"
  fi
  DECAY_STEP="$(python -c "print(round($MAX_STEPS*$DECAY_FRACTION))")"
  WANDB_RUN_ID="$RUN_ID-$(date +%Y%m%d%H%M%S)"
  printf 'MAX_STEPS=%s\nDECAY_STEP=%s\nWANDB_RUN_ID=%s\n' "$MAX_STEPS" "$DECAY_STEP" "$WANDB_RUN_ID" > "$RUN_DIR/plan.env"
  echo "Planned: MAX_STEPS=$MAX_STEPS DECAY_STEP=$DECAY_STEP (effective batch $EFFECTIVE_BATCH)"
fi

if [[ -f "$RUN_DIR/DONE" ]]; then
  echo "Training finished at step $(cat "$RUN_DIR/DONE") (train L1 below $STOP_L1); nothing to do."
  exit 0
fi
if [[ -f "$RUN_DIR--${MAX_STEPS}_chkpt/training_state.pt" ]]; then
  echo "Final checkpoint $RUN_DIR--${MAX_STEPS}_chkpt already exists; nothing to do."
  exit 0
fi

schedule_args=()
if [[ "$AUTO_SCHEDULE" == "1" ]]; then
  schedule_args=(--auto_lr_decay True --stop_l1_threshold "$STOP_L1")
fi

# A STOP left behind by a killed job would end this one immediately
rm -f "$RUN_DIR/STOP" "$RUN_DIR/.timer_stop"
job_start="${SLURM_JOB_START_TIME:-$(date +%s)}"
stop_after=$(( job_start + (JOB_MINUTES - STOP_MARGIN_MINUTES) * 60 - $(date +%s) ))
( sleep "$stop_after" && touch "$RUN_DIR/.timer_stop" "$RUN_DIR/STOP" ) >/dev/null 2>&1 &
timer_pid=$!
trap 'kill "$timer_pid" 2>/dev/null || true' EXIT
echo "Will stop and save in $((stop_after / 60)) min (job limit ${JOB_MINUTES} min)."

# Every job appends to the same W&B run; log steps continue from the resumed step
export WANDB_RUN_ID WANDB_RESUME=allow

resume_args=()
if [[ -f "$RESUME_DIR/training_state.pt" ]]; then
  resume_args=(--resume_from "$RESUME_DIR")
  echo "Resuming from $RESUME_DIR"
fi

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/finetune.py "${common_args[@]}" \
  --run_id_override "$RUN_ID" ${resume_args[@]+"${resume_args[@]}"} \
  --num_steps_before_decay "$DECAY_STEP" --max_steps "$MAX_STEPS" \
  --use_val_set True --val_freq "$SAVE_FREQ" \
  --save_freq "$SAVE_FREQ" --save_latest_checkpoint_only False --merge_lora_during_training "$MERGE_ON_CLUSTER" \
  --resume_save_freq "$RESUME_SAVE_FREQ" ${schedule_args[@]+"${schedule_args[@]}"} ${EXTRA_ARGS:-} \
  2>&1 | tee -a "$RUN_DIR/train.log"

if [[ -f "$RUN_DIR/.timer_stop" && ! -f "$RUN_DIR/DONE" && "$AUTO_RESUBMIT" == "1" ]]; then
  rm -f "$RUN_DIR/.timer_stop"
  # Same GPU count as this job unless NEXT_GPUS asks for another; partition/account/QoS come from #SBATCH lines
  next_gpus="$NUM_GPUS"
  if [[ -f "$RUN_DIR/NEXT_GPUS" ]]; then
    requested="$(tr -dc '0-9' < "$RUN_DIR/NEXT_GPUS")"
    case "$requested" in
      1|2|4|8) next_gpus="$requested" ;;
      *) echo "Ignoring NEXT_GPUS=$requested (use 1, 2, 4 or 8)" ;;
    esac
  fi
  resubmit_args=(--export=ALL --gres="gpu:$next_gpus" --ntasks="$((16 * next_gpus))")
  echo "Time limit reached; resubmitting: sbatch ${resubmit_args[*]} $SCRIPT_PATH"
  sbatch "${resubmit_args[@]}" "$SCRIPT_PATH"
fi
