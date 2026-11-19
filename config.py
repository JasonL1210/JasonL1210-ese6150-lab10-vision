"""Configuration loading and hardware-placeholder protection.

No submission value may be inferred from the simulation configuration.  The
hardware guard is intentionally usable with only the Python standard library;
YAML loading becomes available after PyYAML is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


PENDING_HARDWARE = "PENDING_HARDWARE"


class HardwareConfigurationPending(RuntimeError):
    """Raised when hardware mode is requested before real values are measured."""


RUNTIME_HARDWARE_FIELDS: tuple[str, ...] = (
    "camera.device",
    "camera.intrinsics.fx",
    "camera.intrinsics.fy",
    "camera.intrinsics.cx",
    "camera.intrinsics.cy",
    "lidar.scan_topic",
    "lidar.camera_center_in_lidar_m.x",
    "lidar.camera_center_in_lidar_m.y",
    "lidar.camera_center_in_lidar_m.z",
    "tensorrt.engine_path",
)


SUBMISSION_HARDWARE_FIELDS: tuple[str, ...] = (
    "benchmark.fp32_inference_ms",
    "benchmark.fp16_inference_ms",
    "evidence.lidar_projection_image",
    "evidence.detection_image",
    "evidence.measurement_image",
    "evidence.package_distance_m",
    "evidence.tape_distance_m",
    "evidence.video_link",
)


def is_pending_hardware(value: Any) -> bool:
    """Return whether a field is missing or still explicitly hardware-pending."""

    return value is None or (
        isinstance(value, str) and value.strip().upper() == PENDING_HARDWARE
    )


def _read_dotted(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def pending_fields(
    config: Mapping[str, Any], required_fields: Sequence[str]
) -> list[str]:
    """List required dotted paths that do not contain measured values."""

    return [
        field
        for field in required_fields
        if is_pending_hardware(_read_dotted(config, field))
    ]


def require_hardware_runtime(config: Mapping[str, Any]) -> None:
    """Reject hardware execution until every runtime field has been measured."""

    missing = pending_fields(config, RUNTIME_HARDWARE_FIELDS)
    if missing:
        joined = ", ".join(missing)
        raise HardwareConfigurationPending(
            f"Hardware mode is blocked; measure these fields first: {joined}"
        )


def require_hardware_submission(config: Mapping[str, Any]) -> None:
    """Reject final submission reporting while evidence fields are pending."""

    missing = pending_fields(config, SUBMISSION_HARDWARE_FIELDS)
    if missing:
        joined = ", ".join(missing)
        raise HardwareConfigurationPending(
            f"Submission evidence is incomplete: {joined}"
        )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, with a clear setup message when PyYAML is absent."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise RuntimeError(
            "PyYAML is not installed. Use the Python 3.10/3.11 software environment."
        ) from exc

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return loaded


PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "simulation.yaml"
HARDWARE_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware.yaml"

