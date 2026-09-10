# SO-101 RLDS 1.1.0: Trim → Convert → Verify → Transfer Log

Work log for producing `so101_block_into_cup_50_v5` RLDS version `1.1.0` from the LeRobot source dataset, and
moving it to the AIGS server. Dated 2026-09-10.

## 1. Why 1.1.0 exists

The `1.0.0` conversion included dead time at the start of every episode (the robot sitting still before the
meaningful pick-and-place motion begins). `1.1.0` trims that stationary prefix per episode so training data
starts closer to the first meaningful action, without touching the original LeRobot source dataset.

## 2. Trim onset analysis

A per-episode dynamics analysis produced `outputs/so101-rlds-dynamics/meaningful_action_onsets.csv`, with one row
per episode:

- `episode_index`, `num_frames` — identifies the episode and its original (untrimmed) length.
- `meaningful_onset_zero_based` / `meaningful_onset_one_based` — the frame where real motion starts.
- `stationary_prefix_frames`, `seconds_at_30fps` — how much dead time precedes it.
- `suggested_trim_keep_10_preframes` — the column actually used for trimming: `meaningful_onset` minus a 10-frame
  safety margin, so the first action chunk still contains a few pre-motion frames instead of cutting exactly at
  the onset.

This CSV is the single source of truth for how much each episode gets trimmed; it's referenced (and hashed) from
the conversion manifest so a later conversion can be checked against the same trim decisions.

## 3. Conversion changes (`experiments/robot/so101/convert_lerobot_to_rlds.py`)

- Bumped `VERSION` from `1.0.0` to `1.1.0`.
- Added `--trim-onsets-csv` / `--trim-column` arguments. When given, episode rows with
  `timestamp * fps < trim_start` are dropped before RLDS generation; the source parquet/video files are never
  modified.
- Added stricter validation that used to be implicit:
  - episode metadata parquet loading now supports both "full" metadata (task, video/data file locations) and
    "compact" per-episode length overrides, and rejects duplicate rows for the same episode;
  - trimmed CSV coverage must exactly match the episode indices found in the source metadata (no missing/extra
    episodes);
  - `frame_index` and `timestamp` must be strictly increasing within an episode;
  - action/state arrays must have the expected shape and be finite (no NaN/Inf) after trimming;
  - each episode must still have at least `ACTION_CHUNK_SIZE` (30) frames after trimming;
  - the first retained action chunk must actually contain the annotated meaningful-onset frame (checked against
    `meaningful_onset_zero_based` from the CSV), so an overly aggressive trim can't silently cut past the action
    it was supposed to keep.
- The conversion manifest (`so101_conversion_manifest.json`) now additionally records: a sha256 of the source
  `meta/info.json` and of the trim CSV, the source dataset revision (from the HF cache tree), the dataset version,
  which trim column was used, resolved train/validation episode index lists, and a per-episode record of
  `original_length` / `input_length` / `requested_trim_start` / `effective_first_source_frame` / `output_length`.
  This makes the manifest sufficient on its own to answer "what exactly got cut from episode N" after the fact.

Conversion was run as documented in `SO101.md`:

```bash
python experiments/robot/so101/convert_lerobot_to_rlds.py \
  --source-root /home/yhkim/datasets/so101_block_into_cup_50_v5_trimmed \
  --output-root /home/yhkim/datasets/rlds \
  --validation-episodes 5 \
  --split-seed 7 \
  --image-size 256 \
  --trim-onsets-csv outputs/so101-rlds-dynamics/meaningful_action_onsets.csv \
  --trim-column suggested_trim_keep_10_preframes
```

Result: 50 episodes converted (45 train / 5 validation, seed 7), written to
`/home/yhkim/datasets/rlds/so101_block_into_cup_50_v5/1.1.0/`.

## 4. Verification (`experiments/robot/so101/verify_rlds_dataset.py`)

Added `--expected-version` (fail if the builder directory resolves to a different TFDS version than expected —
guards against silently falling back to the preserved `1.0.0` dataset) and `--full-scan`.

`--full-scan` decodes every episode in both splits and checks, per step:

- image shape/dtype for all three cameras (`front`, `top`, `wrist`) is `(256, 256, 3)` uint8;
- action/state shape matches the 6 joint names and all values are finite;
- `is_first` / `is_last` / `is_terminal` flags are correct on the first step, interior steps, and last step;
- per-episode step count matches `output_length` from the conversion manifest;
- the set of episodes seen in `train` matches `training_episode_indices` and in `validation` matches
  `validation_episode_indices` from the manifest, with no duplicates or unexpected episodes.

Run and result on the lab server (`aga01`):

```text
$ python experiments/robot/so101/verify_rlds_dataset.py \
    --data-root /home/yhkim/datasets/rlds \
    --dataset-name so101_block_into_cup_50_v5 \
    --expected-version 1.1.0 \
    --full-scan

Full scan: 50 episodes / 29651 steps passed
Splits: {'train': 45, 'validation': 5}
Cameras: ('front', 'top', 'wrist')
Joint order: ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos')
Instructions: ['Pick up the block and place it into the cup.']
SO-101 RLDS structural verification passed.
```

`outputs/so101-rlds-dynamics/rlds-1.1.0.sha256` and the `rlds-1.1.0-contact-sheets/` first-frame contact sheets
were kept alongside the CSV as an additional manual/eyeball check of what actually made it into the final dataset.

## 5. Transfer to AIGS

Copied `/home/yhkim/datasets/rlds/so101_block_into_cup_50_v5/1.1.0/` to the AIGS server via `rsync` over SSH (see
[SO101_AIGS_DATASET_TRANSFER.md](SO101_AIGS_DATASET_TRANSFER.md) for the reusable procedure). Re-run
`verify_rlds_dataset.py --full-scan` on the AIGS side against the copied path before using it for training, to
confirm the transfer didn't truncate or corrupt any shard.
