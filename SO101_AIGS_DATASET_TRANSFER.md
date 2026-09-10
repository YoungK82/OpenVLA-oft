# Transferring the SO-101 RLDS Dataset to the AIGS Server

This is the manual for moving a converted SO-101 RLDS dataset from the lab server (`aga01`) to the AIGS cluster
so it is visible to the `openvla-oft` checkout there.

## Prerequisites

- The dataset has already been converted and verified on the lab server (see
  [SO101_RLDS_TRIM_AND_VERIFICATION_LOG.md](SO101_RLDS_TRIM_AND_VERIFICATION_LOG.md) for the 1.1.0 example).
- You have SSH access to the AIGS server, including the non-default port it listens on.
- `rsync` is available on the source side. It is **not** available in a plain Windows PowerShell prompt — run the
  transfer from the lab server (Linux) itself, or from WSL/Git Bash on Windows, not from native PowerShell.

## 1. Locate the dataset on the lab server

RLDS datasets live under:

```text
/home/yhkim/datasets/rlds/<dataset_name>/<version>/
```

Each version directory is self-contained: `dataset_info.json`, `features.json`, `dataset_statistics_*.json`,
`so101_conversion_manifest.json`, and the `.tfrecord` shards for `train`/`validation`. There is nothing else to
collect from the parent directory — copy the version directory as-is.

## 2. Verify before sending

Always run the full-scan verification before transferring, so a bad conversion isn't discovered only after it
reaches AIGS:

```bash
source /home/yhkim/miniconda3/etc/profile.d/conda.sh && conda activate openvla-oft
python experiments/robot/so101/verify_rlds_dataset.py \
  --data-root /home/yhkim/datasets/rlds \
  --dataset-name <dataset_name> \
  --expected-version <version> \
  --full-scan
```

Expect `SO-101 RLDS structural verification passed.` at the end.

## 3. rsync to AIGS

```bash
rsync -avh --partial --progress \
  -e "ssh -p <AIGS_SSH_PORT>" \
  /home/yhkim/datasets/rlds/<dataset_name>/<version>/ \
  aga@<AIGS_HOST>:/gpfsai/home/aga/yhkim/openvla-oft/datasets/rlds/<dataset_name>/<version>/
```

Notes:

- `<AIGS_SSH_PORT>`: AIGS is not reachable on the default port 22 (it will time out); use the 4-digit port from
  your normal `ssh -p ####` login.
- `-e "ssh -p <port>"` is required specifically because rsync otherwise tries the default port.
- `--partial --progress` lets a large transfer resume if interrupted, instead of restarting from scratch.
- The destination path mirrors the same `<dataset_name>/<version>/` layout as the source — this must match
  wherever the AIGS-side `openvla-oft` config expects `RLDS root` to be (see `SO101_INFERENCE_SMOKE_TEST.md`'s
  `RLDS root: /workspace/datasets/rlds` for how the training pod mounts it).

## 4. Verify again on AIGS

Once copied, re-run the same verification on the AIGS side (inside the same conda env, or the pod's
`openvla-oft` environment) against the destination path:

```bash
python experiments/robot/so101/verify_rlds_dataset.py \
  --data-root /gpfsai/home/aga/yhkim/openvla-oft/datasets/rlds \
  --dataset-name <dataset_name> \
  --expected-version <version> \
  --full-scan
```

A pass here confirms the transfer didn't corrupt or truncate any tfrecord shard.
