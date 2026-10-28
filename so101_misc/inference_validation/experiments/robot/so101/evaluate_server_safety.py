"""Offline SO-101 server checks using recorded observations; never connects to a robot.

Dataset bounds are empirical training ranges, not mechanical joint or collision limits.
"""

import argparse
import copy
import json
import time
import uuid
from pathlib import Path

import json_numpy
import numpy as np
import requests
import tensorflow_datasets as tfds

from experiments.robot.so101.constants import JOINT_NAMES
from experiments.robot.so101.evaluate_server_on_rlds import _decode_text, _numpy, _sample_indices
from experiments.robot.so101.so101_utils import validate_and_limit_action

json_numpy.patch()


def query(endpoint, payload, timeout, label):
    payload = copy.deepcopy(payload)
    payload["request_id"] = f"audit-{label}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    record = {"label": label, "request_id": payload["request_id"], "status": response.status_code,
              "round_trip_seconds": elapsed}
    try:
        body = response.json()
        actions = np.asarray(body, dtype=np.float64)
        valid = actions.shape == (30, 6) and np.isfinite(actions).all()
    except (TypeError, ValueError):
        actions, valid = None, False
    record["valid_actions"] = bool(valid)
    if not valid:
        record["response_excerpt"] = response.text[:300]
    return record, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-endpoint", default="http://127.0.0.1:8777/act")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-relative-target", type=float, default=5)
    args = parser.parse_args()
    if args.num_episodes < 1 or args.samples_per_episode < 1 or args.max_relative_target <= 0:
        parser.error("Sample counts and max-relative-target must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())["so101_block_into_cup_50_v5"]
    low, high = np.array(stats["action"]["min"]), np.array(stats["action"]["max"])
    state_low, state_high = np.array(stats["proprio"]["min"]), np.array(stats["proprio"]["max"])
    builder = tfds.builder_from_directory(str(args.builder_dir))
    report = {"builder": str(args.builder_dir.resolve()), "checkpoint": str(args.checkpoint.resolve()),
              "endpoint": args.server_endpoint, "joint_names": list(JOINT_NAMES),
              "max_relative_target": args.max_relative_target, "samples": [], "repeats": [],
              "negative_tests": [], "notes": ["Offline checks do not certify physical robot safety.",
              "Training bounds are not mechanical joint limits; gripper uses its own recorded units."]}
    first_payload, reference = None, None
    errors, jumps, adjacent_jumps, exceeds, bounds, state_bounds, clamp_ok = [], [], [], [], [], [], []
    latencies, predictions = [], []
    for episode in builder.as_dataset(split="validation", shuffle_files=False).take(args.num_episodes):
        steps = list(episode["steps"])
        episode_id = int(_numpy(episode["episode_metadata"]["episode_index"]))
        for frame in _sample_indices(len(steps), args.samples_per_episode):
            step = steps[frame]
            obs = step["observation"]
            images = [_numpy(obs[k]) for k in ("image", "top_image", "wrist_image")]
            state = np.asarray(_numpy(obs["state"]), dtype=np.float32)
            payload = {"full_image": images[0], "additional_images": images[1:], "state": state,
                       "instruction": _decode_text(step["language_instruction"])}
            rec, actions = query(args.server_endpoint, payload, args.timeout, f"ep{episode_id}-frame{frame}")
            rec.update(episode_index=episode_id, frame_index=int(frame), state=state.tolist())
            if not rec["valid_actions"] or rec["status"] != 200:
                report["samples"].append(rec)
                (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
                raise RuntimeError(f"Valid recorded observation failed: {rec}")
            gt = np.stack([_numpy(s["action"]) for s in steps[frame:frame + 30]])
            error = np.abs(actions - gt)
            jump = np.abs(actions[0] - state)
            limited = validate_and_limit_action(actions[0], state, args.max_relative_target)
            clamp_valid = bool(np.isfinite(limited).all() and
                               np.all(np.abs(limited - state) <= args.max_relative_target + 1e-5))
            outside = (actions < low) | (actions > high)
            rec.update(first_action=actions[0].tolist(), first_target_minus_state=(actions[0] - state).tolist(),
                       first_action_abs_error=error[0].tolist(), chunk_mae_per_joint=error.mean(0).tolist(),
                       action_outside_training_range_counts=outside.sum(0).tolist(),
                       limited_first_action=limited.tolist(), clamp_check=clamp_valid)
            report["samples"].append(rec)
            errors.append(error)
            predictions.append(actions)
            jumps.append(jump)
            adjacent_jumps.append(np.abs(np.diff(actions, axis=0)).max(0))
            exceeds.append(jump > args.max_relative_target)
            bounds.append(outside)
            state_bounds.append((state < state_low) | (state > state_high))
            clamp_ok.append(clamp_valid)
            latencies.append(rec["round_trip_seconds"])
            if first_payload is None:
                first_payload, reference = payload, actions
                (args.output_dir / "sample-request.json").write_text(json_numpy.dumps(payload))
            print(f"ep={episode_id} frame={frame} RTT={latencies[-1]:.3f}s finite=(30,6) "
                  f"max_first_delta={jump.max():.3f}", flush=True)

    if first_payload is None:
        raise RuntimeError("No validation samples were evaluated")
    for index in range(3):
        rec, actions = query(args.server_endpoint, first_payload, args.timeout, f"repeat-{index}")
        rec["max_abs_difference_from_first"] = (float(np.max(np.abs(actions - reference)))
                                                if rec["valid_actions"] else None)
        report["repeats"].append(rec)

    cases = {}
    for field in ("instruction", "state"):
        p = copy.deepcopy(first_payload)
        del p[field]
        cases[f"missing_{field}"] = p
    for name, state in (("short_state", np.zeros(1, dtype=np.float32)),
                        ("nan_state", np.full(6, np.nan, dtype=np.float32)),
                        ("inf_state", np.full(6, np.inf, dtype=np.float32))):
        p = copy.deepcopy(first_payload)
        p["state"] = state
        cases[name] = p
    p = copy.deepcopy(first_payload)
    p["additional_images"] = p["additional_images"][:1]
    cases["missing_camera"] = p
    p = copy.deepcopy(first_payload)
    p["full_image"] = p["full_image"].astype(np.float32)
    cases["float_image"] = p
    p = copy.deepcopy(first_payload)
    p["full_image"] = p["full_image"][:, :, :1]
    cases["wrong_image_channels"] = p
    p = copy.deepcopy(first_payload)
    p["instruction"] = ""
    cases["empty_instruction"] = p
    for name, payload in cases.items():
        rec, _ = query(args.server_endpoint, payload, args.timeout, name)
        rec["rejected_with_4xx"] = 400 <= rec["status"] < 500 and not rec["valid_actions"]
        report["negative_tests"].append(rec)
        print(f"negative={name} status={rec['status']} valid_actions={rec['valid_actions']}", flush=True)

    recovery, _ = query(args.server_endpoint, first_payload, args.timeout, "recovery")
    report["recovery"] = recovery
    error_arr, jump_arr, bound_arr = np.array(errors), np.array(jumps), np.array(bounds)
    report["summary"] = {
        "num_samples": len(errors), "num_action_vectors": len(errors) * 30,
        "all_recorded_requests_finite_30x6": True,
        "rtt_seconds": {"min": float(np.min(latencies)), "median": float(np.median(latencies)),
                        "p95": float(np.percentile(latencies, 95)), "max": float(np.max(latencies))},
        "first_action_mae_per_joint": error_arr[:, 0].mean(0).tolist(),
        "chunk_mae_per_joint": error_arr.mean((0, 1)).tolist(),
        "max_first_target_delta_per_joint": jump_arr.max(0).tolist(),
        "max_adjacent_predicted_target_delta_per_joint": np.array(adjacent_jumps).max(0).tolist(),
        "first_targets_exceeding_relative_limit_per_joint": np.array(exceeds).sum(0).tolist(),
        "requests_with_any_first_target_exceeding_relative_limit": int(np.any(exceeds, axis=1).sum()),
        "all_first_target_clamp_checks_passed": all(clamp_ok),
        "action_values_outside_training_range_per_joint": bound_arr.sum((0, 1)).tolist(),
        "max_excess_beyond_training_range_per_joint": np.maximum(
            np.maximum(low - np.array(predictions), np.array(predictions) - high), 0).max((0, 1)).tolist(),
        "predicted_min_per_joint": np.array(predictions).min((0, 1)).tolist(),
        "predicted_max_per_joint": np.array(predictions).max((0, 1)).tolist(),
        "input_states_outside_training_range_per_joint": np.array(state_bounds).sum(0).tolist(),
        "negative_tests_rejected_with_4xx": sum(r["rejected_with_4xx"] for r in report["negative_tests"]),
        "negative_test_count": len(cases),
        "valid_request_after_negative_tests": recovery["status"] == 200 and recovery["valid_actions"],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
