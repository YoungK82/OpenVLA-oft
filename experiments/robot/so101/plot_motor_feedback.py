"""Standalone, read-only Matplotlib viewer for the motor feedback CSV.

No robot, LeRobot, TensorFlow, or network connection is needed. Live mode stays
open when data stops; CSV silence cannot distinguish inference from completion.
"""

import argparse
import csv
import io
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .constants import JOINT_NAMES
else:
    from constants import JOINT_NAMES

MOTORS = tuple(name.removesuffix(".pos") for name in JOINT_NAMES)
METRICS = ("current_A", "load_pct", "torque_est_Nm")
REQUIRED = {"episode", "step", "motor", "timestamp_unix_s", "errors", *METRICS}


class CsvFollower:
    """Read appended bytes only; retain incomplete UTF-8/quoted CSV records.

    Opening briefly on each poll avoids retaining a handle to a replaced file.
    Size reduction or changed file identity resets the reader and plot history.
    Work per live refresh and a malformed record's buffer size are bounded.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.pending = b""
        self.header = None
        self.generation = 0
        self.bad_rows = 0
        self.warning = ""

    def poll(self, limit=262144):
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
                self.offset = 0
                self.pending = b""
                self.header = None
                self.generation += 1
                self.bad_rows = 0
                self.warning = "CSV replaced/truncated; history reset"
            self.identity = identity
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read(limit)
                self.offset += len(chunk)
        except FileNotFoundError:
            self.warning = "Waiting for CSV file"
            return []

        self.pending += chunk
        if chunk and self.warning == "Waiting for CSV file":
            self.warning = ""
        # A physical newline may occur inside the quoted errors field. Only
        # consume a candidate after csv.reader accepts a complete logical row.
        rows = []
        consumed = 0
        scan = 0
        while True:
            end = self.pending.find(b"\n", scan)
            if end < 0:
                break
            scan = end + 1
            candidate = self.pending[consumed:scan]
            try:
                parsed = list(csv.reader(io.StringIO(candidate.decode("utf-8-sig"), newline=""), strict=True))
            except csv.Error as exc:
                if "unexpected end of data" in str(exc):
                    continue
                self.bad_rows += 1
                self.warning = f"Invalid CSV record: {exc}"
                consumed = scan
                continue
            except UnicodeDecodeError:
                self.bad_rows += 1
                self.warning = "Invalid UTF-8 record skipped"
                consumed = scan
                continue
            consumed = scan
            if not parsed or not parsed[0]:
                continue
            values = parsed[0]
            if self.header is None:
                if len(values) != len(set(values)) or not REQUIRED.issubset(values):
                    raise ValueError("CSV header must contain the eight motor-feedback fields")
                self.header = values
            elif len(values) != len(self.header):
                self.bad_rows += 1
                self.warning = "Wrong number of CSV columns; record skipped"
            else:
                rows.append(dict(zip(self.header, values)))
        self.pending = self.pending[consumed:]
        if len(self.pending) > 1048576:
            raise ValueError("Unfinished CSV record exceeds 1 MiB; check the input file")
        return rows


@dataclass
class Sample:
    timestamp: float
    episode: int
    step: int
    values: tuple


class PlotHistory:
    """Keep recent samples in live mode, all samples in offline mode."""

    def __init__(self, window=30.0):
        self.window = window
        self.samples = {motor: deque(maxlen=20000 if window is not None else None) for motor in MOTORS}
        self.first_timestamp = None
        self.latest_timestamp = None
        self.last_received = None
        self.row_count = 0
        self.error_count = 0
        self.latest_error = ""
        self.episode = None
        self.step = None

    def add(self, rows):
        for row in rows:
            errors = []
            try:
                motor = row["motor"]
                if motor not in self.samples:
                    raise ValueError(f"Unknown motor: {motor}")
                timestamp = float(row["timestamp_unix_s"])
                episode, step = int(row["episode"]), int(row["step"])
                if not math.isfinite(timestamp) or episode < 1 or step < 0:
                    raise ValueError("Invalid time/episode/step")
            except (KeyError, TypeError, ValueError) as exc:
                self.error_count += 1
                self.latest_error = str(exc)
                continue
            values = []
            for metric in METRICS:
                value = row.get(metric, "")
                try:
                    number = float(value) if value else math.nan
                    if value and not math.isfinite(number):
                        raise ValueError("non-finite")
                except (TypeError, ValueError):
                    number = math.nan
                    errors.append(f"invalid {metric}")
                values.append(number)
            if row.get("errors"):
                errors.append(row["errors"])
            if errors:
                self.error_count += 1
                self.latest_error = f"{motor}: {'; '.join(errors)}"
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.latest_timestamp = max(timestamp, self.latest_timestamp or timestamp)
            self.samples[motor].append(Sample(timestamp, episode, step, tuple(values)))
            self.episode, self.step = episode, step
            self.row_count += 1
            self.last_received = time.monotonic()
        if self.window is not None and self.latest_timestamp is not None:
            cutoff = self.latest_timestamp - self.window
            for samples in self.samples.values():
                while samples and samples[0].timestamp < cutoff:
                    samples.popleft()

    def series(self, motor, metric_index):
        x, y = [], []
        previous = None
        for sample in self.samples[motor]:
            # Do not join separate episodes, skipped steps, or clock reversals.
            if previous is not None and (sample.episode != previous.episode or
                                         sample.step != previous.step + 1 or
                                         sample.timestamp < previous.timestamp):
                x.append(math.nan)
                y.append(math.nan)
            x.append(sample.timestamp - self.first_timestamp)
            y.append(sample.values[metric_index])
            previous = sample
        return x, y


class FeedbackPlot:
    def __init__(self, reader, *, window=30.0, interval_ms=200, torque=False, offline=False):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, CheckButtons

        self.reader = reader
        self.window = None if offline else window
        self.history = PlotHistory(self.window)
        self.generation = reader.generation
        self.offline = offline
        self.paused = False
        self.animation = None
        self.enabled = {motor: True for motor in MOTORS}
        self.figure, axes = plt.subplots(3 if torque else 2, 1, sharex=True, figsize=(12, 7.5))
        self.axes = list(axes)
        self.figure.subplots_adjust(left=0.09, right=0.77, top=0.86, bottom=0.21, hspace=0.20)
        self.figure.canvas.manager.set_window_title("SO-101 motor feedback")
        self.figure.suptitle("SO-101 motor feedback" + (" | OFFLINE" if offline else " | LIVE"), y=0.97)
        self.figure.text(0.09, 0.91, reader.path.name, fontsize=9)
        self.status = self.figure.text(0.09, 0.115, "Waiting for samples...", fontsize=9)
        self.error_text = self.figure.text(0.09, 0.065, "", fontsize=8, color="firebrick")
        self.figure.text(0.09, 0.025, "Close this window to stop viewer only. CSV is read-only.", fontsize=8)
        labels = ["Current (A)", "PWM load (%)", "Estimated torque (N m)"]
        self.lines = {}
        for index, axis in enumerate(self.axes):
            axis.set_ylabel(labels[index])
            axis.grid(True, alpha=0.25)
            for color, motor in enumerate(MOTORS):
                self.lines[index, motor], = axis.plot([], [], color=f"C{color}", linewidth=1.2, label=motor)
        self.axes[1].set_ylim(-105, 105)
        self.axes[-1].set_xlabel("Elapsed time from first CSV sample (s)")
        selector = self.figure.add_axes((0.79, 0.48, 0.20, 0.32))
        self.checks = CheckButtons(selector, MOTORS, [True] * len(MOTORS))
        for index, label in enumerate(self.checks.labels):
            label.set_color(f"C{index}")
            label.set_fontsize(9)
        self.checks.on_clicked(self.toggle_motor)
        self.pause_button = None
        if not offline:
            self.pause_button = Button(self.figure.add_axes((0.81, 0.36, 0.15, 0.06)), "Pause view")
            self.pause_button.on_clicked(self.toggle_pause)
        if torque:
            self.figure.text(0.79, 0.22, "Torque: uncalibrated\nmagnitude estimate", fontsize=9)
        self.figure.canvas.mpl_connect("close_event", self.close)

        if offline:
            # Full history for review; this mode does not watch future appends.
            while True:
                offset = reader.offset
                self.update()
                if reader.offset == offset:
                    break
            if reader.pending:
                reader.warning = "Unfinished final CSV record ignored"
            self.render()
            self.update_status()
        else:
            self.animation = FuncAnimation(self.figure, self.update, interval=interval_ms,
                                           cache_frame_data=False)

    def update(self, _frame=None):
        try:
            rows = self.reader.poll()
        except OSError as exc:
            self.reader.warning = f"CSV temporarily unavailable: {exc}"
            rows = []
        except ValueError as exc:
            self.reader.warning = str(exc)
            if self.animation is not None:
                self.animation.event_source.stop()
            self.update_status()
            return
        if self.generation != self.reader.generation:
            self.history = PlotHistory(self.window)
            self.generation = self.reader.generation
        self.history.add(rows)
        if not self.paused:
            self.render()
        self.update_status()

    def render(self):
        for (index, motor), line in self.lines.items():
            line.set_data(*self.history.series(motor, index))
            line.set_visible(self.enabled[motor])
        if self.history.latest_timestamp is not None:
            latest = self.history.latest_timestamp - self.history.first_timestamp
            left = 0 if self.offline else max(0, latest - self.window)
            self.axes[-1].set_xlim(left, max(left + 1, latest + 0.1))
        for index, axis in enumerate(self.axes):
            if index == 1:
                continue
            axis.set_autoscaley_on(True)
            axis.relim(visible_only=True)
            axis.autoscale_view(scalex=False, scaley=True)
            # Current and torque are magnitude estimates, never negative axes.
            axis.set_ylim(bottom=0, top=max(0.1, axis.get_ylim()[1]))

    def update_status(self):
        history = self.history
        if history.last_received is None:
            reception = "Waiting for data"
        elif self.offline:
            reception = "Saved CSV (not following)"
        else:
            age = time.monotonic() - history.last_received
            reception = f"Last received {age:.1f}s ago"
            if age >= 2:
                reception += " | No new data (waiting or run ended)"
        mode = "VIEW PAUSED | " if self.paused else ""
        self.status.set_text(
            f"{mode}{reception}\nEpisode {history.episode} | Step {history.step} | "
            f"Rows {history.row_count} | Error rows {history.error_count + self.reader.bad_rows}"
        )
        messages = [s for s in (self.reader.warning, history.latest_error) if s]
        self.error_text.set_text(" | ".join(messages).replace("\n", " ")[:160])

    def toggle_motor(self, motor):
        self.enabled[motor] = not self.enabled[motor]
        for index in range(len(self.axes)):
            self.lines[index, motor].set_visible(self.enabled[motor])
        self.figure.canvas.draw_idle()

    def toggle_pause(self, _event):
        self.paused = not self.paused
        self.pause_button.label.set_text("Resume view" if self.paused else "Pause view")
        if not self.paused:
            self.render()
        self.update_status()
        self.figure.canvas.draw_idle()

    def close(self, _event=None):
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV path; live mode waits if it does not exist yet")
    source.add_argument("--latest", type=Path, metavar="LOG_DIR", help="Select newest MOTOR-FEEDBACK CSV once")
    parser.add_argument("--window", type=positive_float, default=30.0, help="Live history window in seconds")
    parser.add_argument("--interval-ms", type=positive_float, default=200.0, help="Display refresh interval")
    parser.add_argument("--torque", action="store_true", help="Show uncalibrated torque estimate panel")
    parser.add_argument("--offline", action="store_true", help="Show complete saved CSV without following it")
    parser.add_argument("--save", type=Path, help="Offline only: save a figure and exit without a GUI")
    args = parser.parse_args()
    if args.save and not args.offline:
        parser.error("--save requires --offline")
    if args.latest:
        paths = list(args.latest.glob("MOTOR-FEEDBACK-*.csv"))
        if not paths:
            parser.error("No MOTOR-FEEDBACK CSV found; start evaluation first or use --csv")
        args.csv = max(paths, key=lambda path: path.stat().st_mtime_ns)
    if args.offline and not args.csv.is_file():
        parser.error("Offline CSV does not exist")
    print(f"Viewing: {args.csv.resolve()}")
    try:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.exit(1, "Install requirements_motor_plot.txt in the plot environment first.\n")
    plot = FeedbackPlot(CsvFollower(args.csv), window=args.window, interval_ms=args.interval_ms,
                        torque=args.torque, offline=args.offline)
    try:
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            plot.figure.savefig(args.save, dpi=150)
            print(f"Saved: {args.save.resolve()}")
        else:
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plot.close()
        plt.close(plot.figure)


if __name__ == "__main__":
    main()
