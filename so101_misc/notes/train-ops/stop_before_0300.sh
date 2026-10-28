#!/usr/bin/env bash
# Terminates the so101-bidir-b8-s10000 run so GPU 0 is free before 2026-09-29 03:00 KST.
set -u
LOG=/home/yhkim/openvla-oft/outputs/train-logs/so101-bidir-b8-s10000.log
PIDS="2834691 2834766"
start=$(date -d "2026-09-29 02:45:00" +%s)
deadline=$(date -d "2026-09-29 02:55:00" +%s)
now=$(date +%s)
[ "$start" -gt "$now" ] && sleep $((start - now))

alive=""
for pid in $PIDS; do
  grep -q "finetune.py\|torchrun" "/proc/$pid/cmdline" 2>/dev/null && alive="$alive $pid"
done
if [ -z "$alive" ]; then
  echo "$(date) Training already exited; nothing to stop."
  exit 0
fi

# Avoid cutting a checkpoint save short, but never past the deadline.
while [ "$(grep -c 'Saving Model Checkpoint' "$LOG")" -ne "$(grep -c 'Saved merged model' "$LOG")" ] && [ "$(date +%s)" -lt "$deadline" ]; do
  echo "$(date) checkpoint save in progress; waiting..."
  sleep 20
done
if [ "$(grep -c 'Saving Model Checkpoint' "$LOG")" -ne "$(grep -c 'Saved merged model' "$LOG")" ]; then
  echo "$(date) WARNING: deadline reached during a save; the newest checkpoint directory is incomplete."
fi

tr '\r' '\n' < "$LOG" | grep -o "[0-9]*/10000 \[[^]]*\]" | tail -1
kill -TERM $alive 2>/dev/null
for i in $(seq 1 30); do
  still=""
  for pid in $alive; do kill -0 "$pid" 2>/dev/null && still="$still $pid"; done
  [ -z "$still" ] && break
  sleep 2
done
[ -n "$still" ] && { echo "$(date) Forcing kill:$still"; kill -KILL $still 2>/dev/null; sleep 5; }
echo "$(date) Training terminated."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
