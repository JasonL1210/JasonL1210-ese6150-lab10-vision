from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import cv2  # noqa: F401
    import numpy as np

    OPENCV_AVAILABLE = True
except ImportError:
    np = None
    OPENCV_AVAILABLE = False

CALIBRATION_AVAILABLE = OPENCV_AVAILABLE and any(
    (ROOT / "data" / "calibration").glob("*.png")
)


@unittest.skipUnless(
    CALIBRATION_AVAILABLE, "OpenCV and local checkerboard captures are required"
)
class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from calibration import (
            calibrate_directory,
            collect_observations,
            discover_pattern,
            list_images,
            robust_outlier_indices,
        )

        cls.calibrate_directory = staticmethod(calibrate_directory)
        cls.collect_observations = staticmethod(collect_observations)
        cls.discover_pattern = staticmethod(discover_pattern)
        cls.list_images = staticmethod(list_images)
        cls.robust_outlier_indices = staticmethod(robust_outlier_indices)
        cls.image_dir = ROOT / "data" / "calibration"

    def test_all_local_images_have_eight_by_six_corners(self) -> None:
        paths = self.list_images(self.image_dir)
        observations = self.collect_observations(paths, (8, 6))
        self.assertEqual(len(paths), 14)
        self.assertEqual(len(observations), 14)
        self.assertTrue(all(item.image_size == (960, 540) for item in observations))
        self.assertTrue(all(item.corners.shape == (48, 1, 2) for item in observations))

    def test_pattern_discovery_prefers_complete_larger_pattern(self) -> None:
        paths = self.list_images(self.image_dir)[:3]
        pattern = self.discover_pattern(paths, candidates=((7, 5), (8, 6)))
        self.assertEqual(pattern, (8, 6))

    def test_robust_outlier_rule(self) -> None:
        errors = [0.4, 0.45, 0.5, 0.52, 0.55, 0.57, 0.6, 1.5]
        self.assertEqual(self.robust_outlier_indices(errors), (7,))

    def test_real_image_calibration_is_stable_and_auditable(self) -> None:
        run = self.calibrate_directory(self.image_dir, pattern=(8, 6))
        self.assertEqual(len(run.all_observations), 14)
        self.assertEqual(len(run.used_indices), 13)
        self.assertEqual(
            [run.all_observations[index].path.name for index in run.rejected_indices],
            ["14_Color.png"],
        )
        self.assertLess(run.final_fit.rms_px, run.initial_fit.rms_px)
        self.assertLess(run.final_fit.rms_px, 0.6)
        matrix = run.final_fit.camera_matrix
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(650.0 < matrix[0, 0] < 750.0)
        self.assertTrue(650.0 < matrix[1, 1] < 750.0)
        self.assertTrue(430.0 < matrix[0, 2] < 500.0)
        self.assertTrue(240.0 < matrix[1, 2] < 290.0)
        np.testing.assert_allclose(matrix[2], [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

