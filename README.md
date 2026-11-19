# Lab 10 — Independent Local Implementation

This directory is the clean, from-scratch implementation of ESE 6150 Lab 10.
It is intentionally independent of every teammate implementation, checkpoint,
training run, and reported metric.

## Safety boundary

- Everything is developed and tested locally.
- This directory has no Git remote and no code in it performs network writes.
- Hardware-derived values must remain `PENDING_HARDWARE` until measured on the
  actual car and Jetson.
- Simulation values are never valid submission values.

## Current milestone

Milestones **A0** through **A6** are complete. The project layout, interfaces,
configuration boundary, hardware guards, required entry points, independent
single-class RT-DETRv2-R18vd model, post-processing, checkerboard recalibration,
LiDAR/camera geometry, robust depth association, and component/full-model tests
exist. Local COCO-car initialization, F1/10 training, block-separated validation,
checkpoints, metrics, plots, reproducible notebook, checked ONNX graph, dual
ONNX Runtime validation, and Jetson TensorRT build/benchmark tools are complete.
Physical Jetson engine creation remains hardware stage B3 because engines are
target-specific. The offline real-image/synthetic-LiDAR integration is complete.

## Planned stages

1. A0 — local project baseline and test skeleton (complete)
2. A1 — independent RT-DETRv2 implementation and unit tests (complete)
3. A2 — checkerboard recalibration using the local images (complete)
4. A3 — LiDAR/camera geometry and association tests (complete)
5. A4 — local model training and evaluation (complete)
6. A5 — ONNX export, ONNX Runtime comparison, TensorRT source code (complete)
7. A6 — offline integrated simulation (complete)
8. B1–B4 — real camera, real extrinsics, Jetson TensorRT, and real-car evidence

## Python environment

The verified training runtime is the local Python 3.12 `.trainvenv` with
PyTorch 2.6.0+cu124 on the RTX 4060 Laptop GPU.

Run the complete local suite from PowerShell:

```powershell
.\.trainvenv\Scripts\python.exe -m unittest discover -s tests -v
```

The completed training configuration is reproduced by:

```powershell
.\.trainvenv\Scripts\python.exe train_detector.py --epochs 12 --batch-size 16 --gradient-accumulation 1 --warmup-steps 200 --num-workers 2
```

Re-export and validate the fixed-shape deployment graph with:

```powershell
.\.trainvenv\Scripts\python.exe onnx_export.py --provider both --samples 8 --iterations 50
```

See `A5_REPORT.md` for the verified tolerances and `JETSON_TENSORRT.md` for the
target-only FP32/FP16 build and 100-run benchmark commands.

Run the complete offline integration with:

```powershell
.\.trainvenv\Scripts\python.exe integrated.py --mode simulation --provider cpu --frames 8
```

See `A6_REPORT.md` for results and the explicit distinction between real images,
synthetic LiDAR, and still-pending real-car evidence.

The Windows Python 3.14 test discovery remains safe: dependency-specific tests
skip themselves when PyTorch, NumPy, or OpenCV is unavailable.

Later software dependencies are documented in `requirements-software.txt` and
`requirements-training.txt`. Jetson-provided packages are documented in
`requirements-jetson.txt` and must not be installed blindly on the host PC.

