from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interfaces import Detection, FramePacket, LaserScanPacket  # noqa: E402


class InterfaceTests(unittest.TestCase):
    def test_detection_preserves_original_image_coordinates(self) -> None:
        detection = Detection("f1tenth_car", 0.9, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual(detection.label, "f1tenth_car")
        self.assertEqual(detection.box_xyxy, (10.0, 20.0, 110.0, 220.0))

    def test_packets_have_explicit_timestamps_and_frames(self) -> None:
        frame = FramePacket(image=object(), timestamp_s=1.25)
        scan = LaserScanPacket(
            ranges_m=(1.0, 2.0),
            angle_min_rad=-0.1,
            angle_increment_rad=0.1,
            range_min_m=0.05,
            range_max_m=10.0,
            timestamp_s=1.2,
        )
        self.assertEqual(frame.frame_id, "camera")
        self.assertEqual(scan.frame_id, "laser")
        self.assertGreaterEqual(frame.timestamp_s, scan.timestamp_s)


if __name__ == "__main__":
    unittest.main()

