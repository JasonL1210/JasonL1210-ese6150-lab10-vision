from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    HardwareConfigurationPending,
    RUNTIME_HARDWARE_FIELDS,
    SUBMISSION_HARDWARE_FIELDS,
    is_pending_hardware,
    pending_fields,
    require_hardware_runtime,
    require_hardware_submission,
)


class HardwareGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pending_config = {
            "camera": {
                "device": "PENDING_HARDWARE",
                "intrinsics": {
                    "fx": "PENDING_HARDWARE",
                    "fy": "PENDING_HARDWARE",
                    "cx": "PENDING_HARDWARE",
                    "cy": "PENDING_HARDWARE",
                },
            },
            "lidar": {
                "scan_topic": "PENDING_HARDWARE",
                "camera_center_in_lidar_m": {
                    "x": "PENDING_HARDWARE",
                    "y": "PENDING_HARDWARE",
                    "z": "PENDING_HARDWARE",
                },
            },
            "tensorrt": {"engine_path": "PENDING_HARDWARE"},
            "benchmark": {
                "fp32_inference_ms": "PENDING_HARDWARE",
                "fp16_inference_ms": "PENDING_HARDWARE",
            },
            "evidence": {
                "lidar_projection_image": "PENDING_HARDWARE",
                "detection_image": "PENDING_HARDWARE",
                "measurement_image": "PENDING_HARDWARE",
                "package_distance_m": "PENDING_HARDWARE",
                "tape_distance_m": "PENDING_HARDWARE",
                "video_link": "PENDING_HARDWARE",
            },
        }

    def test_pending_marker_and_none_are_rejected(self) -> None:
        self.assertTrue(is_pending_hardware("PENDING_HARDWARE"))
        self.assertTrue(is_pending_hardware(None))
        self.assertFalse(is_pending_hardware(0.0))
        self.assertFalse(is_pending_hardware("/dev/video2"))

    def test_all_runtime_fields_are_pending_in_fresh_config(self) -> None:
        self.assertEqual(
            pending_fields(self.pending_config, RUNTIME_HARDWARE_FIELDS),
            list(RUNTIME_HARDWARE_FIELDS),
        )

    def test_runtime_guard_blocks_fabricated_hardware_execution(self) -> None:
        with self.assertRaises(HardwareConfigurationPending):
            require_hardware_runtime(self.pending_config)

    def test_submission_guard_blocks_missing_evidence(self) -> None:
        self.assertEqual(
            pending_fields(self.pending_config, SUBMISSION_HARDWARE_FIELDS),
            list(SUBMISSION_HARDWARE_FIELDS),
        )
        with self.assertRaises(HardwareConfigurationPending):
            require_hardware_submission(self.pending_config)


if __name__ == "__main__":
    unittest.main()

