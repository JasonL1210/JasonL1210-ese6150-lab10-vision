# Jetson TensorRT handoff

The ONNX graph is portable, but a TensorRT engine is tied to the Jetson GPU,
JetPack, CUDA, and TensorRT versions. Build both engines on the target Jetson;
do not copy an engine produced by another machine.

## 1. Validate the Jetson environment

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Build fixed-shape engines

```bash
python3 convert_trt.py \
  --onnx models/onnx/rtdetrv2_f1tenth_640.onnx \
  --engine models/engines/rtdetrv2_f1tenth_fp32.engine \
  --precision fp32

python3 convert_trt.py \
  --onnx models/onnx/rtdetrv2_f1tenth_640.onnx \
  --engine models/engines/rtdetrv2_f1tenth_fp16.engine \
  --precision fp16
```

Each command validates the parser result, input/output contract, writes a
hardware-specific engine, and records SHA-256 metadata next to the engine.

## 3. Benchmark at least 100 inferences

Use the same Jetson power mode and clocks for both precisions and record them in
the lab report. The script supports TensorRT's name-based v3 API and the older
bindings-based v2 API.

```bash
python3 benchmark_trt.py \
  --engine models/engines/rtdetrv2_f1tenth_fp32.engine \
  --iterations 100 --warmup 20 \
  --output artifacts/benchmark/jetson_fp32.json

python3 benchmark_trt.py \
  --engine models/engines/rtdetrv2_f1tenth_fp16.engine \
  --iterations 100 --warmup 20 \
  --output artifacts/benchmark/jetson_fp16.json
```

Only these target-Jetson measurements may replace the `PENDING_HARDWARE`
latency fields. The Windows ONNX Runtime number is diagnostic and is not a
TensorRT or Jetson result.
