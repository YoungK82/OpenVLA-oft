"""Convert a LeRobot v3 SO-101 dataset into an OpenVLA-compatible RLDS dataset."""

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
from PIL import Image

from experiments.robot.so101.constants import ACTION_CHUNK_SIZE, CAMERA_FIELDS, CONTROL_FREQUENCY, JOINT_NAMES

DATASET_NAME = "so101_block_into_cup_50_v5"


def _read_parquet_files(paths: list[Path]) -> pa.Table:
    if not paths:
        raise FileNotFoundError("No parquet files found")
    return pa.concat_tables([pq.read_table(path) for path in sorted(paths)], promote_options="default")


@dataclass(frozen=True)
class Episode:
    index: int
    length: int
    original_length: int
    task: str
    data_file: Path
    video_files: dict[str, Path]
    video_start_times: dict[str, float]


class SequentialVideoReader:
    """Read monotonically increasing timestamps while retaining only two decoded frames."""

    def __init__(self, path: Path, first_timestamp: float, tolerance: float):
        self.path = path
        self.tolerance = tolerance
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        seek_offset = max(0, round(first_timestamp / self.stream.time_base) - 1)
        self.container.seek(seek_offset, backward=True, any_frame=False, stream=self.stream)
        self.frames = self.container.decode(self.stream)
        self.previous = None
        self.current = self._next_frame()

    def _next_frame(self):
        for frame in self.frames:
            if frame.pts is not None:
                return float(frame.pts * self.stream.time_base), frame
        return None

    def read(self, timestamp: float) -> np.ndarray:
        while self.current is not None and self.current[0] < timestamp:
            self.previous = self.current
            self.current = self._next_frame()

        candidates = [item for item in (self.previous, self.current) if item is not None]
        if not candidates:
            raise RuntimeError(f"No video frame available at {timestamp:.6f}s in {self.path}")
        frame_timestamp, frame = min(candidates, key=lambda item: abs(item[0] - timestamp))
        if abs(frame_timestamp - timestamp) >= self.tolerance:
            raise RuntimeError(
                f"Closest frame in {self.path} is outside tolerance: "
                f"query={timestamp:.6f}, frame={frame_timestamp:.6f}, tolerance={self.tolerance:.6f}"
            )
        return frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        self.container.close()


class So101BlockIntoCup50V5(tfds.core.GeneratorBasedBuilder):
    """SO-101 block-into-cup demonstrations converted from LeRobot v3."""

    name = DATASET_NAME
    pkg_dir_path = Path(__file__).parent
    VERSION = tfds.core.Version("1.1.0")

    def __init__(
        self,
        *args,
        source_root: Path,
        validation_episodes: int,
        split_seed: int,
        image_size: int,
        trim_onsets: dict[int, int] | None,
        **kwargs,
    ):
        self.source_root = Path(source_root)
        self.validation_episodes = validation_episodes
        self.split_seed = split_seed
        self.image_size = image_size
        self.trim_onsets = trim_onsets
        self._info_json = json.loads((self.source_root / "meta" / "info.json").read_text())
        self._validate_source()
        self._episodes = self._load_episodes()
        super().__init__(*args, **kwargs)

    def _validate_source(self) -> None:
        features = self._info_json["features"]
        action_names = tuple(features["action"]["names"])
        state_names = tuple(features["observation.state"]["names"])
        if action_names != JOINT_NAMES or state_names != JOINT_NAMES:
            raise ValueError(f"Unexpected SO-101 joint order: action={action_names}, state={state_names}")
        for camera_key in CAMERA_FIELDS.values():
            if camera_key not in features:
                raise ValueError(f"Missing required camera feature: {camera_key}")

    def _load_episodes(self) -> list[Episode]:
        metadata_paths = sorted((self.source_root / "meta" / "episodes").glob("**/*.parquet"))
        if not metadata_paths:
            raise FileNotFoundError("No episode metadata parquet files found")

        required_columns = {
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            *(f"videos/{camera_key}/chunk_index" for camera_key in CAMERA_FIELDS.values()),
            *(f"videos/{camera_key}/file_index" for camera_key in CAMERA_FIELDS.values()),
            *(f"videos/{camera_key}/from_timestamp" for camera_key in CAMERA_FIELDS.values()),
        }
        full_metadata_tables = []
        current_lengths = {}
        for path in metadata_paths:
            table = pq.read_table(path)
            columns = set(table.column_names)
            if required_columns <= columns:
                full_metadata_tables.append(table)
            elif {"episode_index", "length"} <= columns:
                for row in table.select(["episode_index", "length"]).to_pylist():
                    episode_index = int(row["episode_index"])
                    if episode_index in current_lengths:
                        raise ValueError(f"Duplicate compact metadata for episode {episode_index}")
                    current_lengths[episode_index] = int(row["length"])

        if not full_metadata_tables:
            raise ValueError("No episode metadata file contains task, data, and video locations")
        episodes_table = pa.concat_tables(full_metadata_tables, promote_options="default")
        episodes = []
        seen_indices = set()
        for row in episodes_table.to_pylist():
            episode_index = int(row["episode_index"])
            if episode_index in seen_indices:
                raise ValueError(f"Duplicate full metadata for episode {episode_index}")
            seen_indices.add(episode_index)
            original_length = int(row["length"])
            episode_length = current_lengths.get(episode_index, original_length)
            if not row["tasks"] or not str(row["tasks"][0]).strip():
                raise ValueError(f"Episode {episode_index} has no language task")
            data_chunk = int(row["data/chunk_index"])
            data_file = int(row["data/file_index"])
            videos = {}
            starts = {}
            for source_key in CAMERA_FIELDS.values():
                prefix = f"videos/{source_key}"
                video_chunk = int(row[f"{prefix}/chunk_index"])
                video_file = int(row[f"{prefix}/file_index"])
                videos[source_key] = (
                    self.source_root / "videos" / source_key / f"chunk-{video_chunk:03d}" / f"file-{video_file:03d}.mp4"
                )
                starts[source_key] = float(row[f"{prefix}/from_timestamp"])
            episodes.append(
                Episode(
                    index=episode_index,
                    length=episode_length,
                    original_length=original_length,
                    task=str(row["tasks"][0]),
                    data_file=self.source_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                    video_files=videos,
                    video_start_times=starts,
                )
            )
        episodes = sorted(episodes, key=lambda episode: episode.index)
        episode_indices = {episode.index for episode in episodes}
        if self.trim_onsets is not None and set(self.trim_onsets) != episode_indices:
            missing = sorted(episode_indices - set(self.trim_onsets))
            extra = sorted(set(self.trim_onsets) - episode_indices)
            raise ValueError(f"Trim CSV episode coverage mismatch: missing={missing}, extra={extra}")
        return episodes

    def _info(self) -> tfds.core.DatasetInfo:
        image = tfds.features.Image(
            shape=(self.image_size, self.image_size, 3), dtype=np.uint8, encoding_format="jpeg"
        )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "image": image,
                                "top_image": image,
                                "wrist_image": image,
                                "state": tfds.features.Tensor(shape=(6,), dtype=np.float32),
                            },
                            "action": tfds.features.Tensor(shape=(6,), dtype=np.float32),
                            "language_instruction": tfds.features.Text(),
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                        }
                    ),
                    "episode_metadata": {
                        "episode_index": np.int64,
                        "source_dataset": tfds.features.Text(),
                    },
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        if not 0 < self.validation_episodes < len(self._episodes):
            raise ValueError("validation_episodes must leave at least one episode in each split")
        indices = [episode.index for episode in self._episodes]
        random.Random(self.split_seed).shuffle(indices)
        validation = set(indices[: self.validation_episodes])
        return {
            tfds.Split.TRAIN: self._generate_examples(
                [episode for episode in self._episodes if episode.index not in validation]
            ),
            tfds.Split.VALIDATION: self._generate_examples(
                [episode for episode in self._episodes if episode.index in validation]
            ),
        }

    def _load_episode_rows(self, episode: Episode) -> list[dict]:
        table = pq.read_table(episode.data_file)
        table = table.filter(pc.equal(table["episode_index"], episode.index))
        frame_indices = np.asarray(table["frame_index"])
        if len(frame_indices) > 1 and np.any(np.diff(frame_indices) <= 0):
            raise ValueError(f"Episode {episode.index} frame_index is not strictly increasing")
        rows = table.select(["action", "observation.state", "timestamp", "frame_index"]).to_pylist()
        if len(rows) != episode.length:
            raise ValueError(f"Episode {episode.index} has {len(rows)} rows, expected {episode.length}")
        timestamps = np.asarray([float(row["timestamp"]) for row in rows])
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Episode {episode.index} timestamps are not strictly increasing")
        if self.trim_onsets is not None:
            trim_start = self.trim_onsets[episode.index]
            fps = float(self._info_json["fps"])
            rows = [row for row in rows if round(float(row["timestamp"]) * fps) >= trim_start]
        if len(rows) < ACTION_CHUNK_SIZE:
            raise ValueError(
                f"Episode {episode.index} retains {len(rows)} frames, fewer than the "
                f"{ACTION_CHUNK_SIZE} frames required for one action chunk"
            )
        for row in rows:
            action = np.asarray(row["action"])
            state = np.asarray(row["observation.state"])
            if action.shape != (len(JOINT_NAMES),) or state.shape != (len(JOINT_NAMES),):
                raise ValueError(
                    f"Episode {episode.index} has invalid action/state shape: {action.shape}/{state.shape}"
                )
            if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
                raise ValueError(f"Episode {episode.index} has non-finite action/state values")
        return rows

    def _generate_steps(self, episode: Episode) -> Iterator[dict]:
        rows = self._load_episode_rows(episode)
        tolerance = 1.0 / float(self._info_json["fps"])
        readers = {
            source_key: SequentialVideoReader(
                episode.video_files[source_key],
                episode.video_start_times[source_key] + float(rows[0]["timestamp"]),
                tolerance,
            )
            for source_key in CAMERA_FIELDS.values()
        }
        try:
            for step_index, row in enumerate(rows):
                images = {}
                for output_key, source_key in CAMERA_FIELDS.items():
                    timestamp = episode.video_start_times[source_key] + float(row["timestamp"])
                    frame = readers[source_key].read(timestamp)
                    images[output_key] = np.asarray(
                        Image.fromarray(frame).resize(
                            (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC
                        ),
                        dtype=np.uint8,
                    )
                yield {
                    "observation": {
                        **images,
                        "state": np.asarray(row["observation.state"], dtype=np.float32),
                    },
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "language_instruction": episode.task,
                    "is_first": step_index == 0,
                    "is_last": step_index == len(rows) - 1,
                    "is_terminal": step_index == len(rows) - 1,
                }
        finally:
            for reader in readers.values():
                reader.close()

    def _generate_examples(self, episodes: list[Episode]):
        for episode in episodes:
            yield f"episode_{episode.index:06d}", {
                "steps": self._generate_steps(episode),
                "episode_metadata": {
                    "episode_index": episode.index,
                    "source_dataset": self.source_root.name,
                },
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help="RLDS dataset name; pass the same name to finetune.py --dataset_name. Must contain 'so101'.",
    )
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--trim-onsets-csv", type=Path)
    parser.add_argument("--trim-column", default="suggested_trim_keep_10_preframes")
    parser.add_argument("--expected-fps", type=int, default=CONTROL_FREQUENCY)
    parser.add_argument("--allow-fps-mismatch", action="store_true")
    return parser.parse_args()


def _load_trim_onsets(path: Path, trim_column: str) -> tuple[dict[int, int], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "episode_index" not in reader.fieldnames or trim_column not in reader.fieldnames:
            raise ValueError(
                f"Trim CSV must contain 'episode_index' and {trim_column!r}; found {reader.fieldnames}"
            )
        csv_rows = list(reader)

    trim_onsets = {}
    for row in csv_rows:
        episode_index = int(row["episode_index"])
        trim_start = int(row[trim_column])
        if episode_index in trim_onsets:
            raise ValueError(f"Duplicate trim CSV row for episode {episode_index}")
        if trim_start < 0:
            raise ValueError(f"Episode {episode_index} has negative trim start {trim_start}")
        trim_onsets[episode_index] = trim_start
    return trim_onsets, csv_rows


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.dataset_name) or "so101" not in args.dataset_name:
        raise ValueError(f"--dataset-name must be lowercase snake_case and contain 'so101', got {args.dataset_name!r}")
    So101BlockIntoCup50V5.name = args.dataset_name
    trim_onsets = None
    trim_csv_rows = []
    if args.trim_onsets_csv is not None:
        trim_onsets, trim_csv_rows = _load_trim_onsets(args.trim_onsets_csv, args.trim_column)
    source_info = json.loads((args.source_root / "meta" / "info.json").read_text())
    source_fps = int(source_info["fps"])
    if source_fps != args.expected_fps and not args.allow_fps_mismatch:
        raise ValueError(
            f"Dataset was recorded at {source_fps} FPS, but this recipe uses {args.expected_fps} FPS and "
            f"{ACTION_CHUNK_SIZE}-action chunks. Pass --allow-fps-mismatch only after changing the chunk configuration."
        )
    builder = So101BlockIntoCup50V5(
        data_dir=str(args.output_root),
        source_root=args.source_root,
        validation_episodes=args.validation_episodes,
        split_seed=args.split_seed,
        image_size=args.image_size,
        trim_onsets=trim_onsets,
    )
    trim_csv_by_episode = {int(row["episode_index"]): row for row in trim_csv_rows}
    episode_records = []
    for episode in builder._episodes:
        retained_rows = builder._load_episode_rows(episode)
        first_source_frame = round(float(retained_rows[0]["timestamp"]) * source_fps)
        csv_row = trim_csv_by_episode.get(episode.index)
        if csv_row is not None and csv_row.get("num_frames", "") != "":
            csv_original_length = int(csv_row["num_frames"])
            if csv_original_length != episode.original_length:
                raise ValueError(
                    f"Episode {episode.index} CSV num_frames={csv_original_length}, "
                    f"but full metadata length={episode.original_length}"
                )
        if trim_onsets is not None and trim_onsets[episode.index] >= episode.original_length:
            raise ValueError(
                f"Episode {episode.index} trim start {trim_onsets[episode.index]} is outside "
                f"original length {episode.original_length}"
            )
        record = {
            "episode_index": episode.index,
            "original_length": episode.original_length,
            "input_length": episode.length,
            "requested_trim_start": None if trim_onsets is None else trim_onsets[episode.index],
            "effective_first_source_frame": first_source_frame,
            "output_length": len(retained_rows),
        }
        if csv_row is not None and csv_row.get("meaningful_onset_zero_based", "") != "":
            meaningful_onset = int(csv_row["meaningful_onset_zero_based"])
            if not first_source_frame <= meaningful_onset or meaningful_onset + 2 >= first_source_frame + ACTION_CHUNK_SIZE:
                raise ValueError(
                    f"Episode {episode.index} first action chunk does not contain the three-frame meaningful onset: "
                    f"first={first_source_frame}, onset={meaningful_onset}"
                )
            record["meaningful_onset_zero_based"] = meaningful_onset
        episode_records.append(record)
    builder.download_and_prepare()
    indices = [episode.index for episode in builder._episodes]
    random.Random(args.split_seed).shuffle(indices)
    validation_indices = sorted(indices[: args.validation_episodes])
    training_indices = sorted(indices[args.validation_episodes :])
    source_info_path = args.source_root / "meta" / "info.json"
    revision_files = sorted((args.source_root / ".cache" / "huggingface" / "trees").glob("*.json"))
    manifest = {
        "source_root": str(args.source_root.resolve()),
        "source_info_sha256": hashlib.sha256(source_info_path.read_bytes()).hexdigest(),
        "source_revision": revision_files[0].stem if len(revision_files) == 1 else None,
        "dataset_name": args.dataset_name,
        "source_fps": source_fps,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "joint_names": list(JOINT_NAMES),
        "camera_fields": CAMERA_FIELDS,
        "image_size": args.image_size,
        "dataset_version": str(builder.version),
        "trim_onsets_csv": None if args.trim_onsets_csv is None else str(args.trim_onsets_csv.resolve()),
        "trim_onsets_csv_sha256": (
            None
            if args.trim_onsets_csv is None
            else hashlib.sha256(args.trim_onsets_csv.read_bytes()).hexdigest()
        ),
        "trim_column": args.trim_column if args.trim_onsets_csv is not None else None,
        "validation_episodes": args.validation_episodes,
        "split_seed": args.split_seed,
        "training_episode_indices": training_indices,
        "validation_episode_indices": validation_indices,
        "episodes": episode_records,
    }
    (Path(builder.data_dir) / "so101_conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    split_counts = {name: split.num_examples for name, split in builder.info.splits.items()}
    print(f"RLDS dataset written to: {builder.data_dir}")
    print(f"Splits: {split_counts}")


if __name__ == "__main__":
    main()
