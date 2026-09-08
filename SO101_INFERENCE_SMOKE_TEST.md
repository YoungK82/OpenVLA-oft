# SO-101 Inference Smoke Test on Kubernetes

This runbook validates that the 10-step SO-101 checkpoint can load and process one real RLDS validation observation
end to end. It does **not** connect to or move the physical robot, and it does not measure learned task quality. A
10-step checkpoint is only suitable for plumbing validation.

## Current test resources

```text
Kubernetes namespace:  aga
Pod:                   yhkim-openvla-oft-smoke
GPU node selector:     accelerator=gpuh200
Workspace mount:       /workspace
Conda environment:     /workspace/envs/openvla-oft-cu121
Checkpoint:            /workspace/outputs/so101-smoke-h200
RLDS root:             /workspace/datasets/rlds
Dataset:               so101_block_into_cup_50_v5
Server port:           8777
```

Use two login-node terminals. Terminal A keeps the inference server in the foreground. Terminal B sends the test
request from another process in the same Pod.

## 1. Preflight checks

Run these commands on the login node:

```bash
kubectl -n aga get pod yhkim-openvla-oft-smoke -o wide

kubectl -n aga exec yhkim-openvla-oft-smoke -- bash -lc '
set -e
test -x /workspace/envs/openvla-oft-cu121/bin/python
test -f /workspace/outputs/so101-smoke-h200/dataset_statistics.json
test -f /workspace/outputs/so101-smoke-h200/action_head--latest_checkpoint.pt
test -f /workspace/outputs/so101-smoke-h200/proprio_projector--latest_checkpoint.pt
test -f /workspace/outputs/so101-smoke-h200/model.safetensors.index.json
cd /workspace/code/OpenVLA-oft
/workspace/envs/openvla-oft-cu121/bin/python -c \
  "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__, torch.cuda.get_device_name(0))"
echo "Inference smoke-test preflight passed."
'
```

Do not continue unless the Pod is `Running` and `Ready`, all file checks pass, and the final preflight message is
printed.

## 2. Start the inference server (Terminal A)

Open a shell in the Pod:

```bash
kubectl -n aga exec -it yhkim-openvla-oft-smoke -- bash
```

Inside the Pod, run:

```bash
cd /workspace/code/OpenVLA-oft
export PATH=/workspace/envs/openvla-oft-cu121/bin:$PATH
export HOME=/workspace
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export TRITON_CACHE_DIR=/workspace/cache/triton
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
set -o pipefail

python vla-scripts/deploy.py \
  --robot_platform so101 \
  --pretrained_checkpoint /workspace/outputs/so101-smoke-h200 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key so101_block_into_cup_50_v5 \
  2>&1 | tee /workspace/logs/so101-inference-server.log
```

Model loading can take several minutes. Wait until Uvicorn reports that it is serving on `0.0.0.0:8777`. Keep this
terminal open. A `transformers==4.40.1` versus `4.44.2` warning is already known; record it, but only treat it as a
failure if loading or inference raises an exception.

## 3. Send one RLDS validation observation (Terminal B)

After the server is ready, run this command from a second login-node terminal:

```bash
kubectl -n aga exec yhkim-openvla-oft-smoke -- bash -lc '
set -o pipefail
cd /workspace/code/OpenVLA-oft
export PATH=/workspace/envs/openvla-oft-cu121/bin:$PATH
export HOME=/workspace
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export TRITON_CACHE_DIR=/workspace/cache/triton

python experiments/robot/so101/smoke_test_server.py \
  --data-root /workspace/datasets/rlds \
  --dataset-name so101_block_into_cup_50_v5 \
  --server-endpoint http://127.0.0.1:8777/act \
  --timeout 120 \
  2>&1 | tee /workspace/logs/so101-inference-client.log
'
```

The client reads the first validation step and sends:

```text
front image + top image + wrist image + 6D proprio + language instruction
```

The server must return 30 finite, unnormalized 6D actions.

## 4. Pass criteria

The smoke test passes only when all of these conditions hold:

- Terminal A loads the merged checkpoint, action head, proprio projector, processor, and dataset stats.
- Terminal B exits with status 0 and receives an HTTP success response.
- The client prints `Output actions: shape=(30, 6)`.
- The client prints `SO-101 checkpoint server smoke test passed.`.
- Neither log contains a traceback, CUDA out-of-memory error, `NaN`, or `Inf` action.

Review both logs from the login node:

```bash
tail -100 /home/aga/yhkim/openvla-oft/logs/so101-inference-server.log
tail -100 /home/aga/yhkim/openvla-oft/logs/so101-inference-client.log
```

Also inspect the printed first action and its maximum absolute change from the input state. There is no policy-quality
threshold for this 10-step checkpoint; the values only catch gross unit, normalization, or calibration errors.

## 5. Stop and clean up

Press `Ctrl+C` in Terminal A to stop the server. Verify that no inference process remains:

```bash
kubectl -n aga exec yhkim-openvla-oft-smoke -- \
  bash -lc 'ps -ef | grep "vla-scripts/deploy.py" | grep -v grep || true'
```

Delete the temporary Pod when no more GPU tests are needed:

```bash
kubectl -n aga delete pod yhkim-openvla-oft-smoke --wait=true
```

Deleting the Pod does not delete checkpoint or log files on the PVC.

## Troubleshooting

### `Connection refused`

Terminal A is not ready or exited. Inspect the server log and wait for the Uvicorn startup message before retrying.

### `Action un-norm key ... not found`

Confirm that the checkpoint contains `dataset_statistics.json` and that `--unnorm_key` is exactly
`so101_block_into_cup_50_v5`.

### Missing action head or proprio projector

The checkpoint is incomplete or the wrong directory was passed. The directory must contain exactly one
`action_head` checkpoint and exactly one `proprio_projector` checkpoint.

### CUDA out of memory

Check for another GPU process:

```bash
kubectl -n aga exec yhkim-openvla-oft-smoke -- nvidia-smi
```

Stop unrelated processes only if they belong to this Pod and were started for this test.

### Server returns `error`

Inspect `so101-inference-server.log`; the server logs the underlying traceback. Do not proceed to physical-robot
testing until the offline inference smoke test passes.

## What this test does not prove

Passing confirms the software and data contract from a recorded validation observation to a `(30, 6)` action chunk.
It does not prove that the model can pick up the block, generalize to a new scene, meet real-time latency requirements,
or safely control the robot. Those require a sufficiently trained checkpoint, a physical-robot dry run with motor
commands disabled, and then a supervised rollout with conservative action limits and an emergency stop available.
