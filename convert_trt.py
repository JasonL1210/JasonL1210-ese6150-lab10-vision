"""Build a fixed-shape TensorRT engine from the validated RT-DETRv2 ONNX file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--metadata", type=Path)
    return parser


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace_mib: int = 2048,
) -> dict[str, Any]:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT Python bindings are unavailable. Run this conversion on the target "
            "Jetson using the TensorRT package supplied by JetPack."
        ) from error
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if workspace_mib <= 0:
        raise ValueError("workspace_mib must be positive")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(errors))
    if network.num_inputs != 1 or network.num_outputs != 2:
        raise RuntimeError(
            f"expected one input and two outputs, found {network.num_inputs}/{network.num_outputs}"
        )
    input_tensor = network.get_input(0)
    if tuple(input_tensor.shape) != (1, 3, 640, 640):
        raise RuntimeError(f"unexpected input shape: {tuple(input_tensor.shape)}")
    config = builder.create_builder_config()
    workspace_bytes = workspace_mib * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("target TensorRT platform does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned no serialized engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return {
        "scope": "engine is hardware/TensorRT-version specific",
        "tensorrt_version": trt.__version__,
        "precision": precision,
        "workspace_mib": workspace_mib,
        "input": {"name": input_tensor.name, "shape": list(input_tensor.shape)},
        "outputs": [network.get_output(index).name for index in range(network.num_outputs)],
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "engine": str(engine_path.resolve()),
        "engine_bytes": engine_path.stat().st_size,
        "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_engine(
        args.onnx.resolve(), args.engine.resolve(), args.precision, args.workspace_mib
    )
    metadata = args.metadata or args.engine.with_suffix(args.engine.suffix + ".json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

