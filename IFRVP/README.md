# IFRVP: IFRNet for Video Prediction/Extrapolation

## Overview
IFRVP (IFRNet for Video Prediction) is a PyTorch implementation of a deep learning model for video frame interpolation and extrapolation. The project extends [IFRNet](https://github.com/ltkong218/IFRNet), adapting it for the task of video prediction and frame extrapolation.

## Features
- Multiple model variants (IFRNet, IFRNet_L, IFRNet_S, IFRNet_T, IFRNet_ELAN)
- Training on various datasets (Vimeo90K, GoPro, Cityscapes)
- Distributed training support with PyTorch DDP
- Various loss functions (Charbonnier, Laplacian, Perceptual, Frequency)
- Comprehensive augmentation techniques

## Model Architecture
The architecture consists of:
- Encoder networks that extract multi-scale features
- Decoder networks that generate flow fields and feature maps
- A warping mechanism to align input frames
- Fusion modules to combine aligned features

### Variants
- **IFRNet**: Standard architecture
- **IFRNet_L**: Larger model with more parameters
- **IFRNet_S**: Smaller, more efficient model
- **IFRNet_T**: Tiny model for resource-constrained environments
- **IFRNet_ELAN**: Enhanced model with ELAN (Enhanced Lightweight Attention Network) blocks

## Requirements
- PyTorch
- OpenCV
- NumPy
- imageio
- torchvision
- CUDA-capable GPU (for efficient training)

## Datasets
The model can be trained on:
- **Vimeo90K**: A dataset with 90K video triplets (3 consecutive frames)
- **GoPro**: High frame rate videos for training interpolation models
- **Cityscapes**: Street view driving dataset (extended for temporal prediction)

## Training
The repository provides multiple training scripts:
- `train_vimeo90k.py`: Train on Vimeo90K dataset
- `train_gopro.py`: Train on GoPro dataset
- `train_cityscapes.py` and `train_cityscapes_k+1.py`: Train on Cityscapes for next frame prediction

Example command:
```bash
python -m torch.distributed.launch --nproc_per_node=2 train_vimeo90k.py --model_name IFRNet --batch_size 12
```

### Training Parameters
- `--model_name`: Model architecture to use (default: IFRNet)
- `--epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size per GPU (default: 12)
- `--lr_start`: Initial learning rate (default: 1e-4)
- `--lr_end`: Final learning rate (default: 1e-5)
- `--log_path`: Directory to save checkpoints (default: checkpoint)
- `--resume_epoch`: Resume training from specific epoch (default: 0)
- `--resume_path`: Path to model for resuming training (default: None)

## Loss Functions
The model uses a loss function below:
- **Reconstruction Loss**: L1/Charbonnier loss for pixel-level reconstruction

## Data Augmentation
Various augmentation techniques are implemented:
- Random cropping
- Channel reversing
- Vertical/horizontal flipping
- Random rotation
- Time reversal for temporal robustness

## License
This project is released under the MIT License. See the LICENSE file for details.

## Acknowledgments
This project builds upon IFRNet by Lingtong Kong. The original code has been modified and extended for video prediction tasks.
