from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_required_lab_entry_points_exist(self) -> None:
        for name in ("camera.py", "convert_trt.py", "detection.py", "integrated.py"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file())

    def test_independent_support_files_exist(self) -> None:
        for name in (
            "calibration.py",
            "config.py",
            "interfaces.py",
            "lidar_camera.py",
            "rtdetrv2_model.py",
            "onnx_export.py",
            "benchmark_trt.py",
            "A5_REPORT.md",
            "JETSON_TENSORRT.md",
            "A6_REPORT.md",
        ):
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file())

    def test_local_model_and_artifact_directories_exist(self) -> None:
        for relative in (
            "models/checkpoints",
            "models/onnx",
            "models/engines",
            "artifacts/calibration",
            "artifacts/benchmark",
            "artifacts/hardware_pending",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_dir())

    def test_no_group_repository_remote_metadata(self) -> None:
        git_config = ROOT / ".git" / "config"
        if git_config.is_file():
            contents = git_config.read_text(encoding="utf-8").lower()
            self.assertNotIn("roboracer-class", contents)
            self.assertNotIn("group-12", contents)
            self.assertNotIn("group12", contents)


if __name__ == "__main__":
    unittest.main()

