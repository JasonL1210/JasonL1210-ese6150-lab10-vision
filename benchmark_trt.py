"""Benchmark a TensorRT engine on its target Jetson using CUDA events."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Sequence


def benchmark(engine_path: Path, warmup: int = 20, iterations: int = 100) -> dict:
    if warmup < 1 or iterations < 100:
        raise ValueError("use at least one warmup and at least 100 timed iterations")
    try:
        import tensorrt as trt
        import torch
    except ImportError as error:
        raise RuntimeError("TensorRT and CUDA-enabled PyTorch are required on the Jetson") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("could not deserialize TensorRT engine")
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    tensors = {}
    dtype_map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
    }
    if hasattr(trt, "int64"):
        dtype_map[trt.int64] = torch.int64
    if hasattr(engine, "num_io_tensors") and hasattr(context, "execute_async_v3"):
        api = "name_based_v3"
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for name in names:
            shape = tuple(engine.get_tensor_shape(name))
            dtype = dtype_map[engine.get_tensor_dtype(name)]
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            tensor = (
                torch.rand(shape, device="cuda", dtype=dtype)
                if is_input else torch.empty(shape, device="cuda", dtype=dtype)
            )
            tensors[name] = tensor
            context.set_tensor_address(name, tensor.data_ptr())

        def execute() -> None:
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
    else:
        api = "legacy_bindings_v2"
        bindings = [0] * engine.num_bindings
        for index in range(engine.num_bindings):
            name = engine.get_binding_name(index)
            shape = tuple(context.get_binding_shape(index))
            dtype = dtype_map[engine.get_binding_dtype(index)]
            tensor = (
                torch.rand(shape, device="cuda", dtype=dtype)
                if engine.binding_is_input(index)
                else torch.empty(shape, device="cuda", dtype=dtype)
            )
            tensors[name] = tensor
            bindings[index] = tensor.data_ptr()

        def execute() -> None:
            if not context.execute_async_v2(bindings, stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v2 failed")

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            execute()
    stream.synchronize()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            execute()
            end.record(stream)
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return {
        "scope": "measured on the target that deserialized the engine",
        "device": torch.cuda.get_device_name(),
        "tensorrt_version": trt.__version__,
        "execution_api": api,
        "engine": str(engine_path.resolve()),
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "latency_ms": {
            "mean": statistics.fmean(timings),
            "median": statistics.median(timings),
            "min": min(timings),
            "max": max(timings),
            "p95": sorted(timings)[int(0.95 * (len(timings) - 1))],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = benchmark(args.engine.resolve(), args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
