#!/usr/bin/env bash
#SBATCH --job-name=yhkim-crane-oft-proprio-power
#SBATCH --partition=a100_40g
#SBATCH --account=m-gpu
#SBATCH --qos=m-qos
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Magnet crane OpenVLA-OFT fine-tuning on Hawk (A100 40GB): 3 exo cameras + language (FiLM, task only) + x/y/z/load/power proprio.
# Same recipe as train_crane.sh, split into 24 h jobs that chain themselves until MAX_STEPS.
# Hawk rules (m-qos): max 36 h per job, max 4 GPUs at once, 16 cores per GPU (-n = 16 x GPUs).
#
# Usage (login node, from this folder: shipbuilding/b1_with_power, b2_without_power or b3_proprio_power):
#   conda activate ~/yhkim/so101-openvla-oft/env           # CONDA_EXE is passed on to the job
#   PYTHONPATH=$PWD python experiments/robot/crane/verify_rlds_dataset.py --data-root ~/yhkim/datasets/rlds
#   sbatch experiments/robot/crane/slurm_train_crane.sbatch
#   squeue -u $USER; tail -f checkpoints/<DATASET_NAME>/train.log
# Smoke test (interactive, ~30 min): salloc -p a100_40g -A m-gpu --qos=m-qos -N 1 -n 32 --gres=gpu:2 -t 01:00:00,
#   srun --pty bash, then RUN_ID=smoke MAX_STEPS=60 SAVE_FREQ=60 RESUME_SAVE_FREQ=20 AUTO_RESUBMIT=0 bash <this script>;
#   after "Saved resumable state for Step 20" run `touch checkpoints/smoke/STOP`, rerun, and check "Resuming from" + end at 60.
#
# Resume chain (24 h is the per-job wall time, not the training length):
# - Every RESUME_SAVE_FREQ steps finetune.py refreshes <run_dir>--resume_latest (LoRA, heads, FiLM, proprio projector,
#   optimizer, LR schedule, step, auto-decay state). A job finding it passes --resume_from and continues exactly.
# - JOB_MINUTES - STOP_MARGIN_MINUTES after the job start a timer writes STOP; finetune saves and exits, and this
#   script submits itself again (same GPU count, or `echo 4 > <run_dir>/NEXT_GPUS` for 1/2/4) until MAX_STEPS.
# - A crash resubmits at most MAX_CRASH_RETRIES times in a row (<run_dir>/CRASH_COUNT; reset by a clean timed stop),
#   so a repeatable failure such as OOM stops instead of looping. A node failure requeues the job (--requeue).
# - Manual stop: `touch <run_dir>/STOP` (saves, no resubmit). Continue later with the same sbatch command.
# - If compute nodes cannot run sbatch: AUTO_RESUBMIT=0 and chain from the login node with
#   `sbatch --dependency=afterany:<jobid> ...`; jobs after the end exit at once.
#
# Schedule: effective batch 8 = BATCH_SIZE x GPUs x accumulation (~18.8K steps per epoch; 25K steps ~ 1.33 epochs),
# constant LR 5e-4 by default (LR_DECAY=0); LR_DECAY=1 adds one 10x decay when train L1 plateaus (not before
# MIN_DECAY_STEP, at the latest DECAY_STEP). MAX_STEPS cap.
# No train-L1 early stop: validation + checkpoint every SAVE_FREQ; pick the checkpoint by validation (overfitting guard).
# MAX_STEPS/DECAY_STEP are stored in <run_dir>/plan.env (editable; later jobs read it).
# Checkpoints are unmerged (<run_dir>--<step>_chkpt). Copy one to the inference machine without training_state.pt and
# merge there (vla-scripts/merge_lora_weights_and_save.py), then serve_crane.sh / eval_server_on_rlds.py.
# W&B logs offline into <RUN_ROOT>/wandb as one run across jobs; upload with `wandb sync <RUN_ROOT>/wandb/offline-run-*`.
set -euo pipefail

# ---- Site paths (override with `sbatch --export=ALL,VAR=...`) ----
WORK_DIR="${WORK_DIR:-$HOME/yhkim}"
REPO_DIR="${REPO_DIR:-$WORK_DIR/OpenVLA-oft/shipbuilding/b3_proprio_power}"
CONDA_SH="${CONDA_SH:-${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}}"
CONDA_ENV="${CONDA_ENV:-$WORK_DIR/so101-openvla-oft/env}"
BASE_MODEL="${BASE_MODEL:-$WORK_DIR/so101-openvla-oft/models/openvla-7b-47a0ec7}"
DATA_ROOT="${DATA_ROOT:-$WORK_DIR/datasets/rlds}"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/checkpoints}"

# ---- Recipe ----
DATASET_NAME="${DATASET_NAME:-crane_b_magnet_onoff_proprio_power}"
RUN_ID="${RUN_ID:-$DATASET_NAME}"
BATCH_SIZE="${BATCH_SIZE:-1}"                # per GPU; batch 2 runs out of memory on A100 40GB (accumulation is recomputed)
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MAX_STEPS="${MAX_STEPS:-25000}"
LR_DECAY="${LR_DECAY:-0}"                    # 0 = constant LR; 1 = one automatic 10x decay
DECAY_STEP="${DECAY_STEP:-20000}"            # latest allowed decay (LR_DECAY=1)
MIN_DECAY_STEP="${MIN_DECAY_STEP:-10000}"    # no automatic decay before this step (LR_DECAY=1)
[[ "$LR_DECAY" == "1" ]] || DECAY_STEP=1000000000  # milestone never reached
SAVE_FREQ="${SAVE_FREQ:-5000}"               # validation + checkpoint
RESUME_SAVE_FREQ="${RESUME_SAVE_FREQ:-250}"  # rolling resumable state (lose at most this many steps)
# EXTRA_ARGS: extra finetune.py options (word-split)
WANDB_ENTITY="${WANDB_ENTITY:-}"             # empty = the logged-in account's default entity
WANDB_PROJECT="${WANDB_PROJECT:-crane-openvla-oft}"
JOB_MINUTES="${JOB_MINUTES:-1440}"           # must match #SBATCH --time
STOP_MARGIN_MINUTES="${STOP_MARGIN_MINUTES:-25}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
MAX_CRASH_RETRIES="${MAX_CRASH_RETRIES:-2}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_DIR/experiments/robot/crane/slurm_train_crane.sbatch}"

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "conda.sh not found; submit from a shell where conda works, or pass CONDA_SH=<conda base>/etc/profile.d/conda.sh" >&2
  exit 1
fi
# conda's (de)activate hooks read unset variables
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
cd "$REPO_DIR"
# The env's editable install points at so101-openvla-oft; PYTHONPATH makes this folder's prismatic win
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
echo "Job ${SLURM_JOB_ID:-local} (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname): $DATASET_NAME"
echo "GPUs=$NUM_GPUS per-GPU batch=$BATCH_SIZE accumulation=$GRAD_ACCUM effective batch=$EFFECTIVE_BATCH"
RESUME_DIR="$RUN_DIR--resume_latest"
mkdir -p "$RUN_DIR"

# Stock transformers makes sdpa causal. Refuse to train.
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

if [[ -f "$RUN_DIR/plan.env" ]]; then
  source "$RUN_DIR/plan.env"
  echo "Using stored plan: MAX_STEPS=$MAX_STEPS DECAY_STEP=$DECAY_STEP"
else
  WANDB_RUN_ID="$RUN_ID-$(date +%Y%m%d%H%M%S)"
  printf 'MAX_STEPS=%s\nDECAY_STEP=%s\nWANDB_RUN_ID=%s\n' "$MAX_STEPS" "$DECAY_STEP" "$WANDB_RUN_ID" > "$RUN_DIR/plan.env"
  echo "Planned: MAX_STEPS=$MAX_STEPS DECAY_STEP=$DECAY_STEP (effective batch $EFFECTIVE_BATCH)"
fi

if [[ -f "$RUN_DIR--${MAX_STEPS}_chkpt/training_state.pt" ]]; then
  echo "Final checkpoint $RUN_DIR--${MAX_STEPS}_chkpt already exists; nothing to do."
  exit 0
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
wandb_args=()
[[ -n "$WANDB_ENTITY" ]] && wandb_args=(--wandb_entity "$WANDB_ENTITY")

set +e
# --standalone rendezvous uses the FQDN, which on hawk03 resolves to unreachable nhmgmt.cluster.com; stay on loopback
torchrun --nnodes 1 --nproc-per-node "$NUM_GPUS" --master-addr 127.0.0.1 --master-port $(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 )) \
  vla-scripts/finetune.py \
  --robot_platform crane --vla_path "$BASE_MODEL" \
  --data_root_dir "$DATA_ROOT" --dataset_name "$DATASET_NAME" --run_root_dir "$RUN_ROOT" --run_id_override "$RUN_ID" \
  --use_l1_regression True --use_diffusion False \
  --use_film True --use_proprio True --num_images_in_input 3 \
  --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate "$LEARNING_RATE" --lr_warmup_steps 0 \
  --auto_lr_decay "$( [[ "$LR_DECAY" == "1" ]] && echo True || echo False )" --min_steps_before_auto_decay "$MIN_DECAY_STEP" \
  --num_steps_before_decay "$DECAY_STEP" --max_steps "$MAX_STEPS" \
  --use_val_set True --val_freq "$SAVE_FREQ" --save_freq "$SAVE_FREQ" --save_latest_checkpoint_only False \
  --resume_save_freq "$RESUME_SAVE_FREQ" ${resume_args[@]+"${resume_args[@]}"} \
  --merge_lora_during_training False --image_aug True --lora_rank 32 \
  --wandb_project "$WANDB_PROJECT" ${wandb_args[@]+"${wandb_args[@]}"} ${EXTRA_ARGS:-} \
  2>&1 | tee -a "$RUN_DIR/train.log"
train_status=$?
set -e

resubmit=0
if [[ -f "$RUN_DIR/.timer_stop" && $train_status -eq 0 ]]; then
  rm -f "$RUN_DIR/.timer_stop" "$RUN_DIR/CRASH_COUNT"
  echo "Time limit reached; state saved."
  resubmit=1
elif [[ $train_status -ne 0 ]]; then
  crashes=$(( $(cat "$RUN_DIR/CRASH_COUNT" 2>/dev/null || echo 0) + 1 ))
  echo "$crashes" > "$RUN_DIR/CRASH_COUNT"
  echo "Training exited with status $train_status (crash $crashes of at most $MAX_CRASH_RETRIES retries in a row)." >&2
  (( crashes <= MAX_CRASH_RETRIES )) && resubmit=1
else
  rm -f "$RUN_DIR/CRASH_COUNT"
  echo "Training finished or stopped by hand; not resubmitting."
fi

if [[ $resubmit -eq 1 && "$AUTO_RESUBMIT" == "1" ]]; then
  # Same GPU count as this job unless NEXT_GPUS asks for another; partition/account/QoS come from #SBATCH lines
  next_gpus="$NUM_GPUS"
  if [[ -f "$RUN_DIR/NEXT_GPUS" ]]; then
    requested="$(tr -dc '0-9' < "$RUN_DIR/NEXT_GPUS")"
    case "$requested" in
      1|2|4) next_gpus="$requested" ;;
      *) echo "Ignoring NEXT_GPUS=$requested (use 1, 2 or 4)" ;;
    esac
  fi
  resubmit_args=(--export=ALL --gres="gpu:$next_gpus" --ntasks="$((16 * next_gpus))")
  echo "Resubmitting: sbatch ${resubmit_args[*]} $SCRIPT_PATH"
  sbatch "${resubmit_args[@]}" "$SCRIPT_PATH"
fi
exit "$train_status"
