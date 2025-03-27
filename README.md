# IFRVP: Real-Time Video Prediction with Fast Video Interpolation Model

## Overview

This repository contains the implementation of "REAL-TIME VIDEO PREDICTION WITH FAST VIDEO INTERPOLATION MODEL AND PREDICTION TRAINING," a novel approach to enable zero-latency interaction in networked video applications.

## Paper Link

The full paper is available at: [IEEE Xplore](https://ieeexplore.ieee.org/document/10647865)

## Paper Abstract

Transmission latency significantly affects users' quality of experience in real-time interaction and actuation. While latency is fundamentally inevitable due to physical constraints, this work proposes IFRVP (Intermediate Feature Refinement Video Prediction) to mitigate latency through efficient video prediction. IFRVP extends a simple convolution-only frame interpolation network based on IFRNet by unifying optical flow estimation and pixel refinements into a single network. The architecture introduces ELAN-based residual blocks which significantly reduce computational complexity while maintaining high prediction accuracy. Unlike previous state models that require recursive application for multi-frame prediction, IFRVP's arbitrary and independent prediction methods can generate predictions for any future timestep in a single inference, effectively avoiding error accumulation while enabling real-time performance even on consumer hardware.

## Key Contributions

1. **Three Training Methods for Video Prediction**:
   - **Recurrent Prediction**: Uses the two latest frames to recursively predict the next frame
   - **Arbitrary Prediction**: Predicts any future timestep in a single inference using timestep embedding
   - **Independent Prediction**: Utilizes specialized models for different prediction timesteps

2. **ELAN-based Residual Blocks**: Lightweight architecture that improves both inference speed and prediction accuracy

3. **State-of-the-Art Performance**: Achieves the best trade-off between prediction accuracy and computational speed compared to existing methods

## Results

- IFRVP-Fast achieves comparable or better prediction quality than state-of-the-art methods while requiring only 9.9 GFLOPs (20% less computation than DMVFN)
- Models can run at 70-130 FPS on consumer GPUs depending on resolution
- Independent prediction training achieves the highest accuracy by avoiding error accumulation

## Demo

A demonstration video showing real-time prediction capabilities is available at http://bit.ly/IFRVPDemo

## Applications

- Remote control/telepresence systems
- Autonomous driving
- Cloud gaming
- Video conferencing
- Mission-critical systems requiring near-zero latency

## Citation

```
@inproceedings{hirose2024realtime,
  title={Real-Time Video Prediction with Fast Video Interpolation Model and Prediction Training},
  author={Hirose, Shota and Kotoyori, Kazuki and Arunruangsirilert, Kasidis and Lin, Fangzheng and Sun, Heming and Katto, Jiro},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2024}
}
```
