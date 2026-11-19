"""A6 offline integration regression tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2

from camera import LocalImageSource
from config import SIMULATION_CONFIG_PATH, load_yaml
from simulation.offline_pipeline import DEFAULT_MODEL, run_offline_pipeline
from training_data import load_dataset_index


LOCAL_ASSETS_AVAILABLE = (
    DEFAULT_MODEL.is_file()
    and (Path(__file__).resolve().parents[1] / "data" / "f110_dataset_20220209"
         / "f110_dataset_20220209" / "labels.npy").is_file()
)


@unittest.skipUnless(LOCAL_ASSETS_AVAILABLE, "local dataset and ONNX model are required")
class LocalSourceTests(unittest.TestCase):
    def test_local_source_is_finite_and_timestamped(self) -> None:
        paths, _ = load_dataset_index()
        source = LocalImageSource(paths[:2], fps=20)
        first, second = source.read(), source.read()
        self.assertEqual(first.timestamp_s, 0.0)
        self.assertAlmostEqual(second.timestamp_s, 0.05)
        self.assertEqual(first.image.shape, (360, 640, 3))
        self.assertIsNone(source.read())


@unittest.skipUnless(LOCAL_ASSETS_AVAILABLE, "local dataset and ONNX model are required")
class OfflinePipelineTests(unittest.TestCase):
    def test_real_images_onnx_and_synthetic_lidar_end_to_end(self) -> None:
        config = load_yaml(SIMULATION_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run_offline_pipeline(
                config, DEFAULT_MODEL, output, frame_count=2, provider="cpu"
            )
            self.assertEqual(report["scope"], "offline_real_images_with_synthetic_lidar_only")
            self.assertFalse(report["hardware_submission_evidence"])
            self.assertEqual(report["detection_success_count"], 2)
            self.assertEqual(report["association_success_count"], 2)
            self.assertLess(report["distance_error_m"]["max"], 0.01)
            self.assertLessEqual(report["maximum_sync_skew_s"], 0.0041)
            self.assertTrue((output / "offline_integration_montage.png").is_file())
            video = cv2.VideoCapture(str(output / "offline_integration.mp4"))
            self.assertTrue(video.isOpened())
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            video.release()


if __name__ == "__main__":
    unittest.main()
