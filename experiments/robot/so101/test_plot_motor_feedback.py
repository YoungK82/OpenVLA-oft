"""Read-only viewer checks with synthetic CSVs; never connects to a robot."""

import csv
import io
import math
import tempfile
import unittest
from pathlib import Path

from experiments.robot.so101.plot_motor_feedback import CsvFollower, MOTORS, PlotHistory

FIELDS = ["episode", "step", "motor", "timestamp_unix_s", "current_A", "load_pct", "errors", "torque_est_Nm"]


def make_row(step=0, **kwargs):
    row = dict(episode="1", step=str(step), motor=MOTORS[0], timestamp_unix_s=str(1000 + step),
               current_A="0.65", load_pct="-30", errors="", torque_est_Nm="0.701175475")
    row.update(kwargs)
    return row


def encode(rows, header=False):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, FIELDS)
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="motor-plot-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "feedback.csv"
        self.reader = CsvFollower(self.path)

    def test_partial_multiline_utf8_record_and_no_duplicate_reads(self):
        self.path.write_bytes(encode([], header=True))
        self.assertEqual(self.reader.poll(), [])
        record = encode([make_row(errors='통신 오류, "timeout"\nsecond line')])
        all_rows = []
        for byte in record:
            with self.path.open("ab") as stream:
                stream.write(bytes([byte]))
            all_rows.extend(self.reader.poll())
        self.assertEqual(len(all_rows), 1)
        self.assertEqual(all_rows[0]["errors"], '통신 오류, "timeout"\nsecond line')
        self.assertEqual(self.reader.poll(), [])
        self.assertEqual(self.reader.bad_rows, 0)
        self.assertEqual(self.path.read_bytes(), encode([], header=True) + record)

    def test_missing_file_then_arrival(self):
        self.assertEqual(self.reader.poll(), [])
        self.path.write_bytes(encode([make_row()], header=True))
        self.assertEqual(len(self.reader.poll()), 1)
        self.assertEqual(self.reader.warning, "")

    def test_truncation_resets_reader(self):
        self.path.write_bytes(encode([make_row(step=i) for i in range(5)], header=True))
        self.assertEqual(len(self.reader.poll()), 5)
        self.path.write_bytes(encode([make_row(episode="2")], header=True))
        rows = self.reader.poll()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episode"], "2")
        self.assertEqual(self.reader.generation, 1)

    def test_bad_header_and_wrong_column_count(self):
        self.path.write_bytes(b"wrong,header\n")
        with self.assertRaisesRegex(ValueError, "header"):
            self.reader.poll()
        self.reader = CsvFollower(self.path)
        self.path.write_bytes(encode([], header=True) + b"wrong,count\n" + encode([make_row()]))
        self.assertEqual(len(self.reader.poll()), 1)
        self.assertEqual(self.reader.bad_rows, 1)


class HistoryTests(unittest.TestCase):
    def test_missing_measurements_and_episode_gaps(self):
        history = PlotHistory(window=None)
        history.add([make_row(), make_row(1, current_A="", torque_est_Nm="", errors="current timeout"),
                     make_row(episode="2", timestamp_unix_s="1002")])
        x, y = history.series(MOTORS[0], 0)
        self.assertEqual(y[0], 0.65)
        self.assertTrue(math.isnan(y[1]))
        self.assertTrue(math.isnan(x[2]))
        self.assertTrue(math.isnan(y[2]))
        self.assertEqual(history.error_count, 1)
        _, load = history.series(MOTORS[0], 1)
        self.assertEqual(load[1], -30)

    def test_live_window_prunes_all_motors_and_preserves_time_origin(self):
        history = PlotHistory(window=3)
        history.add([make_row(motor=MOTORS[1])])
        history.add([make_row(step=i) for i in range(10)])
        self.assertEqual(len(history.samples[MOTORS[0]]), 4)
        self.assertEqual(len(history.samples[MOTORS[1]]), 0)
        self.assertEqual(history.series(MOTORS[0], 0)[0], [6, 7, 8, 9])

    def test_bad_values_and_skipped_steps(self):
        history = PlotHistory()
        history.add([make_row(timestamp_unix_s="nan"), make_row(motor="unknown"),
                     make_row(current_A="inf"), make_row(2)])
        self.assertEqual(history.error_count, 3)
        x, y = history.series(MOTORS[0], 0)
        self.assertTrue(math.isnan(y[0]))
        self.assertTrue(math.isnan(x[1]))


class PlotTests(unittest.TestCase):
    def test_saved_plot_controls_and_csv_unchanged(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from experiments.robot.so101.plot_motor_feedback import FeedbackPlot

        with tempfile.TemporaryDirectory(prefix="motor-plot-render-") as directory:
            path = Path(directory) / "demo.csv"
            content = encode([make_row(step=i, motor=motor) for i in range(3) for motor in MOTORS], header=True)
            path.write_bytes(content)
            plot = FeedbackPlot(CsvFollower(path), offline=True, torque=True)
            try:
                self.assertEqual(len(plot.axes), 3)
                self.assertEqual(plot.history.row_count, 18)
                plot.toggle_motor(MOTORS[0])
                self.assertFalse(plot.lines[0, MOTORS[0]].get_visible())
                plot.figure.savefig(Path(directory) / "plot.png")
                self.assertGreater((Path(directory) / "plot.png").stat().st_size, 1000)
                self.assertEqual(path.read_bytes(), content)
                # A later larger measurement must expand the axis after set_ylim.
                plot.history.add([make_row(3, motor=MOTORS[1], current_A="4.0")])
                plot.render()
                self.assertGreater(plot.axes[0].get_ylim()[1], 4.0)
            finally:
                plot.close()
                plt.close(plot.figure)


if __name__ == "__main__":
    unittest.main()
