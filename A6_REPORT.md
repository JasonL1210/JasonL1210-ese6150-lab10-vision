# A6 — Offline integrated perception

## Pipeline exercised

The executable path is:

```text
held-out 640x360 F1/10 image
  → ONNX Runtime RT-DETRv2 detector
  → original-image xyxy detection
  → timestamped synthetic LaserScan
  → LiDAR-to-camera projection
  → box gating and range-discontinuity clustering
  → nearest valid cluster
  → interpolated LiDAR hit and camera-frame 3D position
  → distance, annotated frame, video, and JSON report
```

The eight images come from validation blocks that were not used for training.
Their dataset indices are `250, 285, 421, 706, 742, 1427, 1613, 1649`.
The detector does not use labels or ground-truth boxes in this pipeline.

## Synthetic depth scene

The course dataset contains images and labels but no synchronized LaserScan.
For each detected box, A6 therefore creates a deterministic synthetic scene
containing:

- a target vertical surface at a known depth;
- a background wall;
- a nearer off-axis distractor;
- 1.5 mm standard-deviation range noise;
- 1% random beam dropout;
- up to 4 ms camera/scan timestamp skew.

The synthetic target is positioned behind the detected box center so the test
can focus on projection, gating, cluster rejection, interpolation, coordinate
transforms, and distance calculation. This is not an independent detector
accuracy test; detector accuracy was measured in A4.

## Results

| Measurement | Result |
| --- | ---: |
| Frames | 8 |
| ONNX detections | 8/8 |
| LiDAR associations | 8/8 |
| Detection score range | 0.9601–0.9859 |
| Maximum synchronization skew | 0.004 s |
| Mean distance error | 0.001185 m |
| Median distance error | 0.001144 m |
| Maximum distance error | 0.002245 m |

The montage shows detections in green, all visible synthetic LiDAR returns in a
depth colormap, the selected target cluster in white, and the detection center
in yellow. The MP4 contains the same eight 640×360 frames at 2 FPS.

## Evidence boundary

The output JSON sets `hardware_submission_evidence` to `false`, and every image
is watermarked `REAL IMAGE + SYNTHETIC LIDAR`. These results validate software
composition only. Actual synchronized scans, measured extrinsics, physical
distance error, and integrated real-car video remain `PENDING_HARDWARE`.
