# SO-101 training action/state audit — 2026-10-12

Read the actual RLDS 1.1.0 TFRecords: 45 training episodes / 27,387 frames and 5 validation episodes / 2,264 frames. Analysis compares absolute target action[t] minus measured state[t], not successive target differences. Gripper units are not arm degrees.

| Arm joint | Training signed minimum | Training signed maximum | Latest dry-run first prediction delta |
|---|---:|---:|---:|
| shoulder_lift | -19.52° | +12.40° | +54.21° |
| elbow_flex | -16.44° | +22.59° | -38.13° |
| wrist_flex | -19.69° | +15.74° | -10.37° |
| wrist_roll | -11.87° | +10.20° | -11.07° |

The dry-run column is the first request of the latest run, not its 20-request average. The earlier hardware run started with shoulder_lift +35.58°, also beyond every training target-minus-state value.

Across the first ten frames of all 45 training episodes, maximum absolute shoulder_lift delta was 1.49° and elbow delta 5.98°. For the 100 training states nearest the hardware start (RMS distance scaled by each proprio training range), maximum absolute shoulder_lift delta was 5.71°. Nearest-state search is not image/task-phase matching. Dry-run start is less close to recorded states than the hardware start; marginal per-joint ranges do not establish joint-state distribution coverage.

All 50 RLDS episodes' entire state and action sequences exactly match contiguous paired rows in the source trimmed Parquet files. The manifest uses original recording frame coordinates, whereas the trimmed Parquet frame indices are reset; direct manifest-index filtering does not match, but contiguous paired sequence verification matches 50/50. No additional state/action temporal shift was introduced in this conversion. This does not prove original capture timestamps, camera alignment, or motor calibration were correct.

SO-101 dataset transform preserves absolute actions. Training chunk configuration is observation window 1, future action window 29; chunk target starts at action[t], not action[t+29]. Last 29 frames are excluded as chunk starts. The reported signed extrema are unchanged when restricting to valid chunk starts.

Conclusion: large initial shoulder/elbow deltas observed in the dry run are not typical recorded targets; those signed magnitudes do not occur anywhere in the training split. This strengthens investigation of policy prediction / actual-input mismatch, but does not isolate model weights from visual preprocessing or prove a cause of ground contact. No causal checkpoint inference or robot movement was performed.

Reproducible script: `analyze.py`; complete statistics and nearest states: `report.json`.
