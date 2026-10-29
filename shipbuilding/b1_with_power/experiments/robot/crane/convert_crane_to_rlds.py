"""Convert magnet crane episodes (one LeRobot v3 dataset per TID folder) into an OpenVLA-compatible RLDS dataset."""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pyarrow.parquet as pq
import tensorflow_datasets as tfds

from experiments.robot.crane.constants import (
    ACTION_CHUNK_SIZE,
    CAMERA_FIELDS,
    CONTROL_FREQUENCY,
    DATASET_NAME,
    LANGUAGE_MODE,
    MAGNET_NAMES,
    STATE_NAMES,
    VALIDATION_PER_DATE,
)
from experiments.robot.crane.crane_utils import LANGUAGE_MODES, instruction_text


@dataclass(frozen=True)
class Episode:
    tid: str
    date: str
    length: int
    task: str
    data_file: Path
    video_files: dict[str, Path]


def _video_path(root: Path, meta: dict, source_key: str) -> Path:
    prefix = f"videos/{source_key}"
    return (
        root / "videos" / source_key
        / f"chunk-{int(meta[f'{prefix}/chunk_index']):03d}" / f"file-{int(meta[f'{prefix}/file_index']):03d}.mp4"
    )


def load_episode(folder: Path) -> Episode:
    """Read one TID folder and check it is a single, complete, frame-aligned episode."""
    info = json.loads((folder / "meta" / "info.json").read_text())
    if int(info["fps"]) != CONTROL_FREQUENCY:
        raise ValueError(f"{folder.name}: recorded at {info['fps']} fps, expected {CONTROL_FREQUENCY}")
    features = info["features"]
    if tuple(features["action"]["names"]) != MAGNET_NAMES:
        raise ValueError(f"{folder.name}: unexpected action names {features['action']['names']}")
    if tuple(features["observation.state"]["names"]) != STATE_NAMES:
        raise ValueError(f"{folder.name}: unexpected state names {features['observation.state']['names']}")
    for source_key in CAMERA_FIELDS.values():
        if source_key not in features:
            raise ValueError(f"{folder.name}: missing camera {source_key}")

    episode_rows = []
    for path in sorted((folder / "meta" / "episodes").glob("**/*.parquet")):
        episode_rows += pq.read_table(path).to_pylist()
    if len(episode_rows) != 1:
        raise ValueError(f"{folder.name}: expected one episode per TID folder, found {len(episode_rows)}")
    meta = episode_rows[0]
    if not meta["tasks"] or not str(meta["tasks"][0]).strip():
        raise ValueError(f"{folder.name}: no language task")
    data_file = (
        folder / "data" / f"chunk-{int(meta['data/chunk_index']):03d}" / f"file-{int(meta['data/file_index']):03d}.parquet"
    )
    videos = {}
    for source_key in CAMERA_FIELDS.values():
        if float(meta[f"videos/{source_key}/from_timestamp"]) != 0.0:
            raise ValueError(f"{folder.name}: {source_key} does not start at 0 s; frames would not align with rows")
        videos[source_key] = _video_path(folder, meta, source_key)
    for path in (data_file, *videos.values()):
        if not path.is_file():
            raise FileNotFoundError(f"{folder.name}: missing or unreadable {path}")
    length = int(meta["length"])
    if length < ACTION_CHUNK_SIZE:
        raise ValueError(f"{folder.name}: {length} frames, fewer than one {ACTION_CHUNK_SIZE}-action chunk")
    return Episode(
        tid=folder.name,
        date=folder.name[:8],
        length=length,
        task=str(meta["tasks"][0]),
        data_file=data_file,
        video_files=videos,
    )


def parse_per_date(text: str) -> dict[str, int]:
    """'20261012:1,20261013:6' -> {'20261012': 1, '20261013': 6}"""
    counts = {}
    for item in text.split(","):
        date, count = item.split(":")
        counts[date.strip()] = int(count)
    return counts


def split_by_date(episodes: list[Episode], counts: dict[str, int], seed: int) -> set[str]:
    """Pick `counts[date]` validation TIDs at random within each date; every date keeps training episodes."""
    by_date = defaultdict(list)
    for episode in episodes:
        by_date[episode.date].append(episode.tid)
    if unknown := set(counts) - set(by_date):
        raise ValueError(f"Validation counts name dates with no episodes: {sorted(unknown)}")
    counts = {date: counts.get(date, 0) for date in by_date}
    rng = random.Random(seed)
    validation = set()
    for date in sorted(by_date):
        if counts[date] >= len(by_date[date]):
            raise ValueError(f"Date {date} would have no training episodes")
        validation |= set(rng.sample(sorted(by_date[date]), counts[date]))
    return validation


class CraneMagnetOnoff(tfds.core.GeneratorBasedBuilder):
    """Magnet crane plate-moving episodes: three exo cameras, 4D state (+ crane_pos = x/y/z), 10 magnet on/off commands."""

    name = DATASET_NAME
    pkg_dir_path = Path(__file__).parent
    VERSION = tfds.core.Version("1.0.0")

    def __init__(
        self, *args, episodes: list[Episode], validation_tids: set[str], image_size: int, language_mode: str, **kwargs
    ):
        self._episodes = episodes
        self.language_mode = language_mode
        self.validation_tids = validation_tids
        self.image_size = image_size
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        image = tfds.features.Image(shape=(self.image_size, self.image_size, 3), dtype=np.uint8, encoding_format="jpeg")
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                **{key: image for key in CAMERA_FIELDS},
                                "state": tfds.features.Tensor(shape=(len(STATE_NAMES),), dtype=np.float32),
                                # Proprio input: crane position x, y, z (= state[1:]); the load goes into the language
                                "crane_pos": tfds.features.Tensor(shape=(len(STATE_NAMES) - 1,), dtype=np.float32),
                                # Raw value kept (NaN = power stage off); the instruction text carries it in `power` mode
                                "magnet_power_level": tfds.features.Tensor(shape=(1,), dtype=np.float32),
                            },
                            "action": tfds.features.Tensor(shape=(len(MAGNET_NAMES),), dtype=np.float32),
                            "language_instruction": tfds.features.Text(),
                            "episode_task": tfds.features.Text(),
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                        }
                    ),
                    "episode_metadata": {"tid": tfds.features.Text(), "date": tfds.features.Text()},
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        return {
            tfds.Split.TRAIN: self._generate_examples([e for e in self._episodes if e.tid not in self.validation_tids]),
            tfds.Split.VALIDATION: self._generate_examples([e for e in self._episodes if e.tid in self.validation_tids]),
        }

    def _load_rows(self, episode: Episode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = pq.read_table(
            episode.data_file, columns=["frame_index", "action", "observation.state", "observation.magnet_power_level"]
        )
        if table.num_rows != episode.length:
            raise ValueError(f"{episode.tid}: {table.num_rows} rows, metadata length {episode.length}")
        if not np.array_equal(np.asarray(table["frame_index"]), np.arange(episode.length)):
            raise ValueError(f"{episode.tid}: frame_index is not 0..{episode.length - 1}")
        actions = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
        states = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        if not np.all(np.isin(actions, (0.0, 1.0))):
            raise ValueError(f"{episode.tid}: actions are not all 0/1")
        if not np.all(np.isfinite(states)):
            raise ValueError(f"{episode.tid}: non-finite state values")
        power = table["observation.magnet_power_level"].to_numpy(zero_copy_only=False)
        power = np.asarray([np.nan if v is None else np.asarray(v, dtype=np.float32).ravel()[0] for v in power], np.float32)
        return actions, states, power[:, None]

    def _frames(self, path: Path, expected: int) -> Iterator[np.ndarray]:
        """Decode every frame in order (the videos are constant 30 fps and start at row 0), resized to the RLDS size."""
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            count = 0
            for frame in container.decode(stream):
                count += 1
                if count > expected:
                    raise ValueError(f"{path}: more than {expected} frames")
                yield frame.to_ndarray(
                    width=self.image_size, height=self.image_size, format="rgb24", interpolation="BICUBIC"
                )
            if count != expected:
                raise ValueError(f"{path}: {count} frames, expected {expected}")

    def _generate_steps(self, episode: Episode) -> Iterator[dict]:
        actions, states, power = self._load_rows(episode)
        frames = {key: self._frames(episode.video_files[src], episode.length) for key, src in CAMERA_FIELDS.items()}
        last = episode.length - 1
        for step_index in range(episode.length):
            yield {
                "observation": {
                    **{key: next(frames[key]) for key in CAMERA_FIELDS},
                    "state": states[step_index],
                    "crane_pos": states[step_index, 1:],
                    "magnet_power_level": power[step_index],
                },
                "action": actions[step_index],
                "language_instruction": instruction_text(
                    episode.task, float(states[step_index, 0]), float(power[step_index, 0]), self.language_mode
                ),
                "episode_task": episode.task,
                "is_first": step_index == 0,
                "is_last": step_index == last,
                "is_terminal": step_index == last,
            }
        for generator in frames.values():  # Runs the frame-count check at the end of each video
            if next(generator, None) is not None:
                raise ValueError(f"{episode.tid}: video longer than its {episode.length} rows")

    def _generate_examples(self, episodes: list[Episode]):
        for episode in episodes:
            yield episode.tid, {
                "steps": self._generate_steps(episode),
                "episode_metadata": {"tid": episode.tid, "date": episode.date},
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Folder holding one TID folder per episode")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help="RLDS dataset name; pass the same name to finetune.py --dataset_name. Must contain 'crane'.",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGE_MODES,
        default=LANGUAGE_MODE,
        help="load_power: task + this frame's load and magnet power level; task: the episode task only",
    )
    parser.add_argument(
        "--validation-per-date",
        type=parse_per_date,
        default=VALIDATION_PER_DATE,
        help="Validation episodes per date, picked at random within the date, e.g. 20261012:1,20261013:6,20261014:3",
    )
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.dataset_name) or "crane" not in args.dataset_name:
        raise ValueError(f"--dataset-name must be lowercase snake_case and contain 'crane', got {args.dataset_name!r}")
    CraneMagnetOnoff.name = args.dataset_name

    folders = sorted(path for path in args.source_root.iterdir() if path.is_dir())
    episodes, problems = [], []
    for folder in folders:
        try:
            episodes.append(load_episode(folder))
        except (OSError, ValueError, KeyError, StopIteration) as error:
            problems.append(f"{folder.name}: {error}")
    if problems:
        raise SystemExit("Fix these episode folders before converting:\n  " + "\n  ".join(problems))
    validation_tids = split_by_date(episodes, args.validation_per_date, args.split_seed)

    builder = CraneMagnetOnoff(
        data_dir=str(args.output_root),
        episodes=episodes,
        validation_tids=validation_tids,
        image_size=args.image_size,
        language_mode=args.language,
    )
    builder.download_and_prepare()

    manifest = {
        "source_root": str(args.source_root.resolve()),
        "dataset_name": args.dataset_name,
        "dataset_version": str(builder.version),
        "fps": CONTROL_FREQUENCY,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "magnet_names": list(MAGNET_NAMES),
        "state_names": list(STATE_NAMES),
        "camera_fields": CAMERA_FIELDS,
        "image_size": args.image_size,
        "language_mode": args.language,
        "split_seed": args.split_seed,
        "validation_per_date": args.validation_per_date,
        "validation_tids": sorted(validation_tids),
        "training_tids": sorted(e.tid for e in episodes if e.tid not in validation_tids),
        "episodes": [
            {
                "tid": e.tid,
                "date": e.date,
                "length": e.length,
                "task": e.task,
                "data_sha256": hashlib.sha256(e.data_file.read_bytes()).hexdigest(),
            }
            for e in episodes
        ],
    }
    builder_dir = Path(builder.data_dir)
    (builder_dir / "crane_conversion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    per_date = defaultdict(lambda: [0, 0])
    for e in episodes:
        per_date[e.date][e.tid in validation_tids] += 1
    print(f"RLDS dataset written to: {builder_dir}")
    print(f"Splits: { {name: split.num_examples for name, split in builder.info.splits.items()} }")
    print("Train/validation per date: " + ", ".join(f"{d} {t}/{v}" for d, (t, v) in sorted(per_date.items())))


if __name__ == "__main__":
    main()
