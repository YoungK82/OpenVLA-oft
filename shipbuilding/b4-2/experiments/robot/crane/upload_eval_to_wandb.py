"""Add a --val_only evaluation (e.g. CKPT_DIR/test_metrics_stride3.json from val_crane.sh EVAL_SPLIT=test) to the W&B run of
the training run that saved the checkpoint, as "VLA Test/*" metrics plotted against `checkpoint_step`.

Run with a Python that has a logged-in wandb (the training env's wandb 0.16.6 is not logged in; we use a separate venv):
  WANDB_SYNC_ENTITY=<entity> ~/wandb-cli/bin/python experiments/robot/crane/upload_eval_to_wandb.py CKPT_DIR/test_metrics_stride3.json
The W&B run is found by its name, "ft+<RUN_ID>" (RUN_ID = the checkpoint folder name without "--<step>_chkpt"), or given
with --run-id. A successful upload leaves <metrics_json>.wandb (run id and step); a JSON that has it is skipped (--force
logs it again). The W&B history is not used for this check: right after an upload it may not show the new row yet.
"""

import argparse
import json
import os
import re
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics_json", type=Path, nargs="+")
    parser.add_argument("--entity", default=os.environ.get("WANDB_SYNC_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_SYNC_PROJECT", "crane-openvla-oft"))
    parser.add_argument("--run-id", default=None, help="W&B run id (default: find the run by name)")
    parser.add_argument("--force", action="store_true", help="log even if this checkpoint step is already logged")
    args = parser.parse_args()
    if not args.entity:
        raise SystemExit("Set --entity or WANDB_SYNC_ENTITY")
    api = wandb.Api(timeout=60)

    for path in args.metrics_json:
        m = json.loads(path.read_text())
        split = m.get("split", "validation")
        if split == "validation":
            print(f"{path}: validation re-evaluations are not uploaded (training already logs VLA Val/*)")
            continue
        prefix = f"VLA {split.title()}"
        marker = path.with_name(path.name + ".wandb")
        if marker.exists() and not args.force:
            print(f"{path}: already uploaded ({marker.read_text().strip()}); skipped")
            continue
        ckpt_name = Path(m["checkpoint"]).name
        run_name = "ft+" + re.sub(r"--\d+_chkpt$", "", ckpt_name)

        run_id = args.run_id
        if run_id is None:
            runs = sorted(
                api.runs(f"{args.entity}/{args.project}", filters={"display_name": run_name}),
                key=lambda r: r.created_at,
            )
            if not runs:
                raise SystemExit(f"{path}: no W&B run named {run_name!r} in {args.entity}/{args.project}")
            run_id = runs[-1].id
        payload = {
            "checkpoint_step": m["step"],
            f"{prefix}/Loss": m["loss_value"],
            f"{prefix}/Curr Action L1 Loss": m["curr_action_l1_loss"],
            f"{prefix}/Next Actions L1 Loss": m["next_actions_l1_loss"],
            f"{prefix}/Onoff Accuracy": m["onoff_accuracy"],
            f"{prefix}/Mean Balanced Accuracy": m["mean_balanced_accuracy"],
            f"{prefix}/Frames": m["frames"],
        }
        for i, (on, off) in enumerate(zip(m["on_recall"], m["off_recall"]), start=1):
            payload[f"{prefix}/On Recall {i}"] = on
            payload[f"{prefix}/Off Recall {i}"] = off

        run = wandb.init(entity=args.entity, project=args.project, id=run_id, resume="must")
        run.define_metric("checkpoint_step")
        run.define_metric(f"{prefix}/*", step_metric="checkpoint_step")
        run.log(payload)
        run.finish()
        marker.write_text(f"{args.entity}/{args.project}/{run_id} {prefix} checkpoint_step={m['step']}\n")
        print(f"{path}: logged {prefix} at checkpoint_step {m['step']} to run {run_id} ({run_name})")


if __name__ == "__main__":
    main()
