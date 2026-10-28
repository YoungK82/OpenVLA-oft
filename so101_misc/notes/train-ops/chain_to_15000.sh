#!/usr/bin/env bash
# Waits for the 5000->10000 run to finish cleanly, then starts the 10000->15000 run.
set -u
cd /home/yhkim/openvla-oft
LOG=outputs/train-logs/so101-bidir-b8-s10000.log
CKPT=checkpoints/so101-bidir-b8-s10000--10000_chkpt
PID=2895608
echo "$(date) waiting for training PID $PID to exit"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "$(date) PID $PID exited"

if ! grep -q "Max step 10000 reached" "$LOG" || ! grep -q "Saved merged model for Step 10000" "$LOG" \
   || [ ! -f "$CKPT/action_head--10000_checkpoint.pt" ] || [ ! -f "$CKPT/proprio_projector--10000_checkpoint.pt" ] \
   || [ ! -f "$CKPT/model.safetensors.index.json" ]; then
  echo "$(date) 10000-step run did not finish cleanly; NOT starting the 15000 run."
  exit 1
fi

# Another user may grab GPU 0 in between; wait up to 3 h for ~88 GB free.
for i in $(seq 1 180); do
  free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 88000 ] && break
  echo "$(date) GPU 0 free ${free} MiB < 88000; waiting"
  sleep 60
done
if [ "$free" -lt 88000 ]; then echo "$(date) GPU 0 still busy; NOT starting."; exit 1; fi

echo "$(date) starting 10000->15000 run"
exec bash outputs/train-logs/so101-bidir-b8-s15000-resume.sh
