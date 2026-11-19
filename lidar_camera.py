"""LiDAR/camera geometry and detection-depth association.

Frames follow the handout: LiDAR is x-forward/y-left/z-up; camera optical is
x-right/y-down/z-forward. Functions are ROS-independent NumPy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from interfaces import LaserScanPacket


AXIS_SWAP_LIDAR_TO_CAMERA = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=float
)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("focal lengths and image dimensions must be positive")

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float
        )

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height
        )


@dataclass(frozen=True)
class RigidTransform:
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation must be 3x3 and translation length three")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation determinant must be +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points must end in three coordinates")
        return points @ self.rotation.T + self.translation

    def inverse_apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points must end in three coordinates")
        return (points - self.translation) @ self.rotation

    @property
    def target_origin_in_source(self) -> np.ndarray:
        """Camera center in LiDAR coordinates for an L→C transform."""

        return -self.rotation.T @ self.translation


@dataclass(frozen=True)
class ProjectedScan:
    beam_indices: np.ndarray
    angles_rad: np.ndarray
    ranges_m: np.ndarray
    points_lidar: np.ndarray
    points_camera: np.ndarray
    pixels_uv: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.beam_indices)
        arrays = (
            self.angles_rad,
            self.ranges_m,
            self.points_lidar,
            self.points_camera,
            self.pixels_uv,
        )
        if any(len(array) != count for array in arrays):
            raise ValueError("projected scan arrays must have equal lengths")

    def subset(self, indices: np.ndarray) -> "ProjectedScan":
        return ProjectedScan(
            self.beam_indices[indices],
            self.angles_rad[indices],
            self.ranges_m[indices],
            self.points_lidar[indices],
            self.points_camera[indices],
            self.pixels_uv[indices],
        )


@dataclass(frozen=True)
class AssociationConfig:
    trim_fraction: float = 0.15
    column_half_width_px: float | None = 14.0
    vertical_margin_px: float = 2.0
    absolute_range_jump_m: float = 0.20
    relative_range_jump: float = 0.08
    maximum_beam_gap: int = 2
    minimum_cluster_size: int = 2
    method: Literal["depth", "vertical_plane"] = "depth"

    def __post_init__(self) -> None:
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if self.column_half_width_px is not None and self.column_half_width_px <= 0:
            raise ValueError("column_half_width_px must be positive or None")
        if self.minimum_cluster_size < 1:
            raise ValueError("minimum_cluster_size must be positive")


@dataclass(frozen=True)
class AssociationResult:
    box_center_uv: np.ndarray
    hit_lidar: np.ndarray
    hit_camera: np.ndarray
    position_lidar: np.ndarray
    position_camera: np.ndarray
    distance_camera_m: float
    cluster_beam_indices: np.ndarray
    cluster_ranges_m: np.ndarray
    method: str


def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def make_lidar_to_camera_transform(
    camera_center_in_lidar_m: Sequence[float],
    residual_rpy_rad: Sequence[float] = (0.0, 0.0, 0.0),
) -> RigidTransform:
    """Return P_C=R_CL P_L+t_CL; residual RPY acts after the fixed axis swap."""

    center = np.asarray(camera_center_in_lidar_m, dtype=float)
    residual = np.asarray(residual_rpy_rad, dtype=float)
    if center.shape != (3,) or residual.shape != (3,):
        raise ValueError("camera center and residual RPY must each have length three")
    rotation = rotation_matrix_rpy(*residual) @ AXIS_SWAP_LIDAR_TO_CAMERA
    return RigidTransform(rotation, -rotation @ center)


def scan_to_lidar_points(
    scan: LaserScanPacket,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.asarray(scan.ranges_m, dtype=float)
    indices = np.arange(ranges.size, dtype=np.int64)
    angles = scan.angle_min_rad + indices * scan.angle_increment_rad
    valid = (
        np.isfinite(ranges)
        & (ranges >= scan.range_min_m)
        & (ranges <= scan.range_max_m)
    )
    ranges, indices, angles = ranges[valid], indices[valid], angles[valid]
    points = np.column_stack(
        (ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges))
    )
    return indices, angles, ranges, points


def project_camera_points(
    points_camera: np.ndarray, intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape [N,3]")
    positive = points[:, 2] > 0
    pixels = np.full((len(points), 2), np.nan)
    visible = points[positive]
    pixels[positive, 0] = intrinsics.fx * visible[:, 0] / visible[:, 2] + intrinsics.cx
    pixels[positive, 1] = intrinsics.fy * visible[:, 1] / visible[:, 2] + intrinsics.cy
    return pixels, positive


def project_laser_scan(
    scan: LaserScanPacket,
    intrinsics: CameraIntrinsics,
    lidar_to_camera: RigidTransform,
    clip_to_image: bool = True,
) -> ProjectedScan:
    beams, angles, ranges, lidar_points = scan_to_lidar_points(scan)
    camera_points = lidar_to_camera.apply(lidar_points)
    pixels, valid = project_camera_points(camera_points, intrinsics)
    if clip_to_image:
        valid &= (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < intrinsics.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < intrinsics.height)
        )
    return ProjectedScan(
        beams[valid], angles[valid], ranges[valid], lidar_points[valid],
        camera_points[valid], pixels[valid]
    )


def backproject_pixel(
    pixel_uv: Sequence[float], depth_z_m: float, intrinsics: CameraIntrinsics
) -> np.ndarray:
    if not np.isfinite(depth_z_m) or depth_z_m <= 0:
        raise ValueError("optical-axis depth must be positive and finite")
    u, v = map(float, pixel_uv)
    return np.array(
        [
            (u - intrinsics.cx) * depth_z_m / intrinsics.fx,
            (v - intrinsics.cy) * depth_z_m / intrinsics.fy,
            depth_z_m,
        ]
    )


def vertical_plane_intersection(
    pixel_uv: Sequence[float],
    hit_lidar: np.ndarray,
    intrinsics: CameraIntrinsics,
    lidar_to_camera: RigidTransform,
) -> tuple[np.ndarray, np.ndarray]:
    camera_center = lidar_to_camera.target_origin_in_source
    hit = np.asarray(hit_lidar, dtype=float)
    horizontal = hit[:2] - camera_center[:2]
    length = np.linalg.norm(horizontal)
    if length <= 1e-9:
        raise ValueError("hit is too close to the camera center")
    normal = np.array([horizontal[0] / length, horizontal[1] / length, 0.0])
    u, v = map(float, pixel_uv)
    ray_camera = np.array(
        [(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0]
    )
    ray_lidar = lidar_to_camera.rotation.T @ ray_camera
    denominator = float(normal @ ray_lidar)
    if abs(denominator) <= 1e-9:
        raise ValueError("pixel ray is parallel to the vertical surface")
    scale = float(normal @ (hit - camera_center) / denominator)
    if scale <= 0:
        raise ValueError("intersection is behind the camera")
    point_lidar = camera_center + scale * ray_lidar
    return point_lidar, lidar_to_camera.apply(point_lidar)


def split_range_clusters(scan: ProjectedScan, config: AssociationConfig) -> list[np.ndarray]:
    if len(scan.beam_indices) == 0:
        return []
    order = np.argsort(scan.beam_indices)
    clusters: list[list[int]] = [[int(order[0])]]
    for previous, current in zip(order[:-1], order[1:]):
        threshold = max(
            config.absolute_range_jump_m,
            config.relative_range_jump * min(scan.ranges_m[previous], scan.ranges_m[current]),
        )
        should_split = (
            scan.beam_indices[current] - scan.beam_indices[previous] > config.maximum_beam_gap
            or abs(scan.ranges_m[current] - scan.ranges_m[previous]) > threshold
        )
        if should_split:
            clusters.append([])
        clusters[-1].append(int(current))
    return [
        np.asarray(cluster, dtype=np.int64)
        for cluster in clusters
        if len(cluster) >= config.minimum_cluster_size
    ]


def interpolate_lidar_hit(scan: ProjectedScan, target_u: float) -> np.ndarray:
    order = np.argsort(scan.pixels_uv[:, 0])
    columns, points = scan.pixels_uv[order, 0], scan.points_lidar[order]
    if len(points) == 1 or target_u <= columns[0]:
        return points[0].copy()
    if target_u >= columns[-1]:
        return points[-1].copy()
    upper = int(np.searchsorted(columns, target_u, side="right"))
    lower = upper - 1
    span = columns[upper] - columns[lower]
    if abs(span) <= 1e-9:
        return (points[lower] + points[upper]) / 2
    weight = (target_u - columns[lower]) / span
    return points[lower] * (1 - weight) + points[upper] * weight


def associate_detection_to_scan(
    box_xyxy: Sequence[float],
    projected_scan: ProjectedScan,
    intrinsics: CameraIntrinsics,
    lidar_to_camera: RigidTransform,
    config: AssociationConfig = AssociationConfig(),
) -> AssociationResult | None:
    x0, y0, x1, y1 = map(float, box_xyxy)
    if not (x1 > x0 and y1 > y0):
        raise ValueError("detection box must have positive width and height")
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    gate_x0 = x0 + config.trim_fraction * (x1 - x0)
    gate_x1 = x1 - config.trim_fraction * (x1 - x0)
    if config.column_half_width_px is not None:
        gate_x0 = max(gate_x0, center[0] - config.column_half_width_px)
        gate_x1 = min(gate_x1, center[0] + config.column_half_width_px)
    pixels = projected_scan.pixels_uv
    gate = (
        (pixels[:, 0] >= gate_x0)
        & (pixels[:, 0] <= gate_x1)
        & (pixels[:, 1] >= y0 - config.vertical_margin_px)
        & (pixels[:, 1] <= y1 + config.vertical_margin_px)
    )
    candidate_ids = np.flatnonzero(gate)
    if not candidate_ids.size:
        return None
    candidates = projected_scan.subset(candidate_ids)
    clusters = split_range_clusters(candidates, config)
    if not clusters:
        return None
    chosen_ids = min(
        clusters, key=lambda ids: float(np.median(candidates.ranges_m[ids]))
    )
    cluster = candidates.subset(chosen_ids)
    hit_lidar = interpolate_lidar_hit(cluster, center[0])
    hit_camera = lidar_to_camera.apply(hit_lidar)
    if hit_camera[2] <= 0:
        return None
    if config.method == "depth":
        position_camera = backproject_pixel(center, float(hit_camera[2]), intrinsics)
        position_lidar = lidar_to_camera.inverse_apply(position_camera)
    else:
        position_lidar, position_camera = vertical_plane_intersection(
            center, hit_lidar, intrinsics, lidar_to_camera
        )
    return AssociationResult(
        center, hit_lidar, hit_camera, position_lidar, position_camera,
        float(np.linalg.norm(position_camera)), cluster.beam_indices.copy(),
        cluster.ranges_m.copy(), config.method
    )


__all__ = [
    "AXIS_SWAP_LIDAR_TO_CAMERA", "AssociationConfig", "AssociationResult",
    "CameraIntrinsics", "ProjectedScan", "RigidTransform",
    "associate_detection_to_scan", "backproject_pixel", "interpolate_lidar_hit",
    "make_lidar_to_camera_transform", "project_camera_points", "project_laser_scan",
    "rotation_matrix_rpy", "scan_to_lidar_points", "split_range_clusters",
    "vertical_plane_intersection",
]

