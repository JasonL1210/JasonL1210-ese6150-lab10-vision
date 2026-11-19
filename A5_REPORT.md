# A5 — ONNX and TensorRT software completion

## Export contract

- Source checkpoint: selected epoch-12 EMA model
- Input: `images`, float32, `[1, 3, 640, 640]`
- Outputs: `pred_logits [1, 300, 1]`, `pred_boxes [1, 300, 4]`
- ONNX opset: 17; IR version: 8
- Graph: 1,271 nodes and 249 initializers
- File size: 79,831,826 bytes
- SHA-256: `ce4f70d99fab18c36b51f0dac69382a570662a92180c821af3f7dc1fd62c86c5`

The graph is fixed-shape intentionally. The camera pipeline resizes input to
640×640, and a fixed graph gives TensorRT the simplest and most repeatable
optimization target.

## Numerical validation

Eight frames distributed across the leakage-resistant validation blocks were
compared with the fused PyTorch model.

| Runtime | Validation rule | Worst result | Required |
| --- | --- | --- | --- |
| ONNX Runtime CPU | every query, elementwise logits | `9.6321e-5` max abs | `< 1e-3` |
| ONNX Runtime CPU | every query, elementwise boxes | `2.2161e-4` max abs | `< 1e-3` |
| ONNX Runtime CUDA | final top score | `3.3152e-4` abs | `< 0.01` |
| ONNX Runtime CUDA | final top box | `3.2514e-5` max abs | `< 0.005` |
| ONNX Runtime CUDA | PyTorch/ORT top-box IoU | `0.999882` minimum | `> 0.99` |

CUDA reductions can reorder low-confidence decoder queries, so comparing query
index *n* to the same index across accelerators is not semantically meaningful.
The CPU graph still passes the stricter full-tensor check; the CUDA check uses
the actual deployed result: the highest-score detection and its box. All raw
full-query differences remain recorded in the JSON report.

## Local timing boundary

ONNX Runtime CPU averaged `127.70 ms` over 50 runs on the Windows development
PC. This number is diagnostic only. ONNX Runtime CUDA timing is intentionally
omitted because part of this graph falls back to CPU in the Windows provider,
making the result neither a pure GPU measurement nor representative of Jetson
TensorRT.

## TensorRT handoff

`convert_trt.py` parses the graph, verifies its I/O contract, builds either an
FP32 or FP16 engine, rejects FP16 targets without fast FP16 capability, and
writes version/hash metadata. `benchmark_trt.py` supports TensorRT v2 binding
and v3 name-based execution APIs and enforces at least 100 timed iterations.

Actual engine files and latency values remain `PENDING_HARDWARE`; producing
them on Windows or copying an engine from a different device would be invalid.
Run the commands in `JETSON_TENSORRT.md` on the assigned Jetson.
