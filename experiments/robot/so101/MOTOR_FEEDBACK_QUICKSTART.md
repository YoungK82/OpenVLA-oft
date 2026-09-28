# Motor feedback: changes and commands

Run commands from the repository root. Keep the existing robot, camera, server,
task, and dry-run options in your evaluation command.

## What the feature adds

- `run_so101_eval.py` calls `MotorFeedbackLogger` after `get_observation()`
  and before inference/action dispatch on every control step.
- The logger reads `Present_Current` and `Present_Load` in two sequential
  bus transactions, with no retries or register writes.
- These readings are logged only; they are not added to the policy input or
  used to modify actions. Extra bus reads and CSV writes add loop latency;
  30 Hz is a target, not a guaranteed measured sampling frequency.
- Each step writes six CSV rows and flushes them. Failed measurements remain
  blank with an error message; previous values are not reused.
- Current is raw count × 0.0065 A. Load is signed count / 10 (% PWM duty).
  Torque is an uncalibrated magnitude estimate from current and a configurable
  constant (default 11 kgf·cm/A), converted to N·m.
- Inference blocks sampling. The timestamp is the host-side start of feedback
  reads, not an exact simultaneous servo measurement time.
- `plot_motor_feedback.py` runs separately and reads only the CSV.

## Record feedback

Append these options to your existing evaluation command:

```text
--log_motor_feedback True --torque_constant_kgf_cm_per_A 11.0
```

Both values are already defaults. The CSV is written to
`experiments/logs/MOTOR-FEEDBACK-<timestamp><note>.csv` unless
`--local_log_dir` changes the directory. The evaluation terminal prints the path.

To disable feedback CSV creation and the extra bus reads:

```text
--log_motor_feedback False
```

`--dry_run True` still connects to the robot and reads observations/feedback;
it skips action dispatch. It is not the offline plotting mode.

## GUI (second terminal)

```powershell
conda activate openvla-oft
# Install once if needed:
python -m pip install -r experiments/robot/so101/requirements_motor_plot.txt
python experiments/robot/so101/plot_motor_feedback.py --latest experiments/logs --torque
```

Start the viewer after the new evaluation CSV has been created. `--latest`
selects a file once; it does not switch to subsequent runs. Alternatively use
`--csv <path-printed-by-evaluation>` to select the exact run.
The GUI requires a desktop and a Matplotlib GUI backend such as Tk or Qt.
Closing the viewer does not stop the evaluation or CSV recording.

## Existing CSV: no robot connection

```powershell
python experiments/robot/so101/plot_motor_feedback.py --csv experiments/logs/MOTOR-FEEDBACK-2026_09_22-18_05_50--restored-original-ol30.csv --offline --torque
```

Add `--save outputs/motor-feedback.png` to save an image without opening a GUI.

## Files to review for GitHub

Integrate into the target repository and review its diff before committing:

- `experiments/robot/so101/run_so101_eval.py`
- `experiments/robot/so101/motor_feedback.py`
- `experiments/robot/so101/plot_motor_feedback.py`
- `experiments/robot/so101/requirements_motor_plot.txt`
- `experiments/robot/so101/test_motor_feedback.py`
- `experiments/robot/so101/test_plot_motor_feedback.py`
- `experiments/robot/so101/MOTOR_FEEDBACK.md`
- `experiments/robot/so101/MOTOR_PLOT.md`
- `experiments/robot/so101/MOTOR_FEEDBACK_QUICKSTART.md`

This assumes the target already contains the SO-101 client and its dependencies,
including `constants.py`, `so101_utils.py`, and `vla_client_utils.py`.
Do not stage the entire working folder indiscriminately: virtual environments,
bytecode, experiment CSVs, images, and rollout videos are separate artifacts.
The evaluation script retains the existing guard against more than three
open-loop hardware actions. Disabling that guard is not part of this feature.

## Offline tests

```powershell
python -B -m unittest experiments.robot.so101.test_motor_feedback experiments.robot.so101.test_plot_motor_feedback -v
```

Tests use mock hardware and saved/synthetic CSVs; they do not validate physical
robot timing or calibrate torque. See [MOTOR_FEEDBACK.md](MOTOR_FEEDBACK.md) and
[MOTOR_PLOT.md](MOTOR_PLOT.md) for details.
