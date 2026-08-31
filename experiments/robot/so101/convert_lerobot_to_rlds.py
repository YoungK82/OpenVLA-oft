"""Convert a LeRobot v3 SO-101 dataset into an OpenVLA-compatible RLDS dataset."""

import argparse
import json
import random
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

DATASET_NAME = "so101_block_into_cup_50_v5"
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
CAMERA_FIELDS = {
    "image": "observation.images.front",
    "top_image": "observation.images.top",
    "wrist_image": "observation.images.wrist",
}


def _read_parquet_files(paths: list[Path]) -> pa.Table:
    if not paths:
        raise FileNotFoundError("No parquet files found")
    return pa.concat_tables([pq.read_table(path) for path in sorted(paths)], promote_options="default")


@dataclass(frozen=True)
class Episode:
    index: int
    length: int
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
    VERSION = tfds.core.Version("1.0.0")

    def __init__(
        self,
        *args,
        source_root: Path,
        validation_episodes: int,
        split_seed: int,
        image_size: int,
        **kwargs,
    ):
        self.source_root = Path(source_root)
        self.validation_episodes = validation_episodes
        self.split_seed = split_seed
        self.image_size = image_size
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
        episodes_table = _read_parquet_files(list((self.source_root / "meta" / "episodes").glob("**/*.parquet")))
        episodes = []
        for row in episodes_table.to_pylist():
            episode_index = int(row["episode_index"])
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
                    length=int(row["length"]),
                    task=str(row["tasks"][0]),
                    data_file=self.source_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                    video_files=videos,
                    video_start_times=starts,
                )
            )
        return sorted(episodes, key=lambda episode: episode.index)

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
        table = table.sort_by([("frame_index", "ascending")])
        rows = table.select(["action", "observation.state", "timestamp", "frame_index"]).to_pylist()
        if len(rows) != episode.length:
            raise ValueError(f"Episode {episode.index} has {len(rows)} rows, expected {episode.length}")
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
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = So101BlockIntoCup50V5(
        data_dir=str(args.output_root),
        source_root=args.source_root,
        validation_episodes=args.validation_episodes,
        split_seed=args.split_seed,
        image_size=args.image_size,
    )
    builder.download_and_prepare()
    split_counts = {name: split.num_examples for name, split in builder.info.splits.items()}
    print(f"RLDS dataset written to: {builder.data_dir}")
    print(f"Splits: {split_counts}")


if __name__ == "__main__":
    main()
