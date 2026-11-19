from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy tests run in the local WSL environment")
class LidarCameraGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from interfaces import LaserScanPacket
        from lidar_camera import (
            AXIS_SWAP_LIDAR_TO_CAMERA,
            AssociationConfig,
            CameraIntrinsics,
            ProjectedScan,
            associate_detection_to_scan,
            backproject_pixel,
            make_lidar_to_camera_transform,
            project_camera_points,
            project_laser_scan,
            scan_to_lidar_points,
            split_range_clusters,
        )
        from simulation.synthetic_scan import default_demo_scene, generate_scan

        globals().update(
            LaserScanPacket=LaserScanPacket,
            AXIS_SWAP_LIDAR_TO_CAMERA=AXIS_SWAP_LIDAR_TO_CAMERA,
            AssociationConfig=AssociationConfig,
            CameraIntrinsics=CameraIntrinsics,
            ProjectedScan=ProjectedScan,
            associate_detection_to_scan=associate_detection_to_scan,
            backproject_pixel=backproject_pixel,
            make_lidar_to_camera_transform=make_lidar_to_camera_transform,
            project_camera_points=project_camera_points,
            project_laser_scan=project_laser_scan,
            scan_to_lidar_points=scan_to_lidar_points,
            split_range_clusters=split_range_clusters,
            default_demo_scene=default_demo_scene,
            generate_scan=generate_scan,
        )

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(600.0, 600.0, 480.0, 270.0, 960, 540)
        self.transform = make_lidar_to_camera_transform((0.08, 0.0, -0.06))

    def test_axis_swap_matches_handout(self) -> None:
        point = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            AXIS_SWAP_LIDAR_TO_CAMERA @ point, [-2.0, -3.0, 1.0]
        )

    def test_measured_camera_center_maps_to_camera_origin(self) -> None:
        center = np.array([0.08, 0.0, -0.06])
        np.testing.assert_allclose(self.transform.apply(center), np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(self.transform.target_origin_in_source, center)

    def test_project_then_backproject_recovers_camera_point(self) -> None:
        point = np.array([[0.2, -0.1, 2.0]])
        pixels, positive = project_camera_points(point, self.intrinsics)
        self.assertTrue(positive[0])
        recovered = backproject_pixel(pixels[0], point[0, 2], self.intrinsics)
        np.testing.assert_allclose(recovered, point[0], atol=1e-12)

    def test_intrinsics_scale_x_and_y_independently(self) -> None:
        scaled = self.intrinsics.scaled(480, 180)
        self.assertEqual((scaled.fx, scaled.cx), (300.0, 240.0))
        self.assertEqual((scaled.fy, scaled.cy), (200.0, 90.0))

    def test_invalid_scan_ranges_are_filtered(self) -> None:
        scan = LaserScanPacket(
            ranges_m=(np.nan, np.inf, 0.01, 1.0, 20.0),
            angle_min_rad=0.0,
            angle_increment_rad=0.1,
            range_min_m=0.05,
            range_max_m=10.0,
            timestamp_s=0.0,
        )
        beams, _, ranges, points = scan_to_lidar_points(scan)
        np.testing.assert_array_equal(beams, [3])
        np.testing.assert_allclose(ranges, [1.0])
        self.assertEqual(points.shape, (1, 3))

    def test_range_discontinuity_creates_two_clusters(self) -> None:
        scan = ProjectedScan(
            beam_indices=np.array([10, 11, 12, 13]),
            angles_rad=np.zeros(4),
            ranges_m=np.array([2.0, 2.01, 5.0, 5.01]),
            points_lidar=np.zeros((4, 3)),
            points_camera=np.zeros((4, 3)),
            pixels_uv=np.array([[478, 250], [479, 250], [481, 260], [482, 260]]),
        )
        clusters = split_range_clusters(scan, AssociationConfig())
        self.assertEqual(len(clusters), 2)
        np.testing.assert_array_equal(clusters[0], [0, 1])
        np.testing.assert_array_equal(clusters[1], [2, 3])

    def test_synthetic_target_is_selected_in_front_of_wall_and_distractor(self) -> None:
        scan = generate_scan(default_demo_scene(), noise_std_m=0.002, seed=6150)
        projected = project_laser_scan(scan, self.intrinsics, self.transform)
        box = (360.0, 150.0, 600.0, 430.0)
        result = associate_detection_to_scan(
            box,
            projected,
            self.intrinsics,
            self.transform,
            AssociationConfig(column_half_width_px=80.0),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.hit_lidar[0], 2.0, delta=0.01)
        self.assertAlmostEqual(result.hit_lidar[1], 0.0, delta=0.01)
        self.assertAlmostEqual(result.position_camera[2], 1.92, delta=0.01)
        self.assertGreaterEqual(len(result.cluster_beam_indices), 2)
        self.assertTrue(np.median(result.cluster_ranges_m) < 2.1)

    def test_level_camera_depth_and_vertical_plane_methods_agree(self) -> None:
        projected = project_laser_scan(
            generate_scan(default_demo_scene()), self.intrinsics, self.transform
        )
        box = (360.0, 150.0, 600.0, 430.0)
        depth = associate_detection_to_scan(
            box, projected, self.intrinsics, self.transform,
            AssociationConfig(column_half_width_px=80.0, method="depth"),
        )
        plane = associate_detection_to_scan(
            box, projected, self.intrinsics, self.transform,
            AssociationConfig(column_half_width_px=80.0, method="vertical_plane"),
        )
        self.assertIsNotNone(depth)
        self.assertIsNotNone(plane)
        np.testing.assert_allclose(depth.position_camera, plane.position_camera, atol=1e-10)

    def test_vertical_gate_rejects_scan_line_outside_box(self) -> None:
        projected = project_laser_scan(
            generate_scan(default_demo_scene()), self.intrinsics, self.transform
        )
        result = associate_detection_to_scan(
            (360.0, 300.0, 600.0, 500.0),
            projected,
            self.intrinsics,
            self.transform,
            AssociationConfig(column_half_width_px=80.0),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

