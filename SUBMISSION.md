# Lab 10: Vision Lab — Local Completion Tracker

This is a working local tracker, not a fabricated final submission. Replace a
field only after the corresponding software result or hardware measurement is
actually produced.

## Team

Team members: PENDING_USER_INPUT

## Code

| Part | Local file |
| --- | --- |
| Camera | `camera.py` |
| TensorRT conversion | `convert_trt.py` |
| TensorRT benchmark | `benchmark_trt.py` |
| ONNX export and validation | `onnx_export.py` |
| Detection | `detection.py` |
| Integration | `integrated.py` |
| Training notebook | `notebooks/rtdetrv2_f1tenth_from_scratch.ipynb` |

## 1. Accessing Images

Resolution and measured frame rate: PENDING_HARDWARE

## 2. Distance Measurement

Factory intrinsics at 960x540: PENDING_HARDWARE

Camera lens position in LiDAR frame: PENDING_HARDWARE

LiDAR projection calibration image: PENDING_HARDWARE

## 3. Network Training

Epochs, batch size and learning rate: `12`, `16`, `1e-4`; AMP enabled, 200-step warmup

Final training loss: `1.101874` (weighted RT-DETRv2 loss, epoch 12)

Validation at epoch 12: AP50 `1.0000`, mean IoU `0.974041`, thresholded
accuracy `1.0000` over 250 leakage-resistant block-held-out frames.

Training and validation plot: `artifacts/training/training_history.png`

Selected checkpoint: `artifacts/training/selected/model.pt` (epoch 12 EMA)

Validation montage: `artifacts/training/selected_validation_predictions.png`

## 4. TensorRT Deployment

Software handoff complete: fixed `[1,3,640,640]` ONNX graph, opset 17, SHA-256
`ce4f70d99fab18c36b51f0dac69382a570662a92180c821af3f7dc1fd62c86c5`.
Eight-frame CPU elementwise and CUDA top-detection comparisons passed. See
`artifacts/benchmark/onnx_validation.json` and `A5_REPORT.md`.

FP32 and FP16 engine files: PENDING_HARDWARE — must be built on the target
Jetson because TensorRT engines are GPU/JetPack/TensorRT specific.

FP16 inference time over at least 100 measured runs: PENDING_HARDWARE

FP32 inference time over at least 100 measured runs: PENDING_HARDWARE

FP16 comparison and explanation: PENDING_HARDWARE

## 5. Integration

Offline software integration complete: eight held-out real F1/10 images were
processed by the exported ONNX detector and associated with noisy synthetic
LaserScan data containing a target, background wall, dropouts, and an off-axis
distractor. Detection and association succeeded on 8/8 frames; maximum synthetic
distance error was `0.002245 m`. See `A6_REPORT.md`.

Offline montage: `artifacts/simulation/offline_integration/offline_integration_montage.png`

Offline video: `artifacts/simulation/offline_integration/offline_integration.mp4`

These offline artifacts are not real-car submission evidence.

Detection image: PENDING_HARDWARE

Measurement image: PENDING_HARDWARE

Package distance: PENDING_HARDWARE

Tape distance: PENDING_HARDWARE

Video link: PENDING_HARDWARE

## Extra Credit

Recalibrated K from the local 960x540 checkerboard set (preliminary offline result):

```text
[[694.52450551,   0.0,        458.26496194],
 [  0.0,        696.62952851, 263.16013948],
 [  0.0,          0.0,          1.0       ]]
```

RMS reprojection error: `0.499200 px` using 13 views. All 14 views had an
8x6 corner detection; `14_Color.png` was rejected by the robust per-view error
rule. Comparison with factory K: PENDING_HARDWARE.

Other network: NOT_STARTED

LiDAR/depth-image comparison: PENDING_HARDWARE

