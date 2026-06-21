# CSIG-VI


## CSIG-VI — Low-Light Image Enhancement

>`CSIG-VI/`

2025 "Camera Academic Star" Imaging Algorithm Technology Competition entry. Three model variants targeting low-light image enhancement, from unsupervised to fully supervised approaches. Target hardware: RTX 4060 (8GB VRAM), PyTorch 2.0.1 with CUDA 11.7/11.8.

### DarkVisionNet (`DarkvisionNet.py`)

Unsupervised dual-branch architecture with physics-guided attention:

| Component | Architecture |
|---|---|
| `PhysicalGuidedAttention` | Fuses luminance distribution (Conv -> Sigmoid) with noise level estimation (variance pooling) to generate content-aware attention weights |
| `MultiScaleResidualBlock` | Three parallel convolutions (3x3, 5x5, 7x7) -> fusion -> PhysicalGuidedAttention gating -> residual connection. 8 blocks stacked. |
| `EnhancementBranch` | Encoder-decoder: Conv 64 -> Conv 128 (stride 2) -> Conv 256 (stride 2) -> 8x MultiScaleResidualBlock -> ConvTranspose 128 -> ConvTranspose 64 -> Conv 3 (Sigmoid) |
| `DenoisingBranch` | 5-layer Conv-BN-ReLU stack (64 channels, 3x3 kernels), no downsampling — preserves fine detail |
| `DarkVisionNet` | Runs both branches in parallel, concatenates outputs (6 channels), fuses via Conv 32 -> Conv 3 (Tanh), adds residual to input, clamps to [0,1] |

**Training**: Self-supervised (input = target), patch-based (256x256), synthetic darkening augmentation (gamma 1.5-3.0, additive Gaussian noise). Loss = MSE + 0.5*SSIM + 0.1*ColorConsistencyLoss. Adam optimizer with CosineAnnealingLR. 4-day time limit built into training loop.

```
python DarkvisionNet.py --input_dir <path> --epochs 100 --batch_size 8
```

### RD-DualNet (`RD-DualNet.py` / `RD-DualNet2.py`)

Supervised approach — directly learns the mapping from low-light to Ground Truth. Key innovation: **Retinex-DCP Guided Attention** that fuses physical priors with learned features.

| Component | Architecture |
|---|---|
| `DepthwiseSeparableConv` | Depthwise conv (groups=in_channels) + pointwise conv (1x1) + BN + ReLU — mobile-friendly |
| `SEBlock` | Squeeze-and-Excitation channel attention with reduction=16 |
| `RetinexDCPGuidedAttention` | Estimates Retinex reflectance R (detail) via 2-layer Conv and DCP prior (illumination/noise) via 2-layer Conv, fuses with backbone features to produce attention weights |
| `LightweightMultiScaleBlock` | 3 parallel DepthwiseSeparableConv (3x3, 5x5, 7x7) -> SEBlock -> 1x1 fusion -> RetinexDCPGuidedAttention gating -> residual |
| `RD_DualNet` | Shared shallow feature extractor (Conv 64) -> Enhancement branch (4x LightweightMultiScaleBlock, detail/brightness focus) + Denoising branch (3x DepthwiseSeparableConv + SEBlock, smoothness/noise suppression) -> feature concatenation -> Conv 1x1 -> Conv 3x3 (Tanh) -> residual output |

**v2 fixes** (`RD-DualNet2.py`): Replaced `nn.Sequential` with `nn.ModuleList` for proper forward iteration through enhance blocks; added VGG denormalization `[-1,1] -> [0,1]` before perceptual loss computation; added post-training color calibration via least-squares linear regression (`evaluate_and_tune`: learns a 3x3 gain matrix + 3x1 bias from evaluation pairs).

**Training**: Supervised paired dataset (`PairedDataset` — low_dir + high_dir), patch-based (256x256, paired random crops). **HybridLoss** = L1 (pixel fidelity) + 0.1*Perceptual (VGG16 features up to relu4_3) + 0.05*GT-Mean (global brightness alignment). AdamW (lr=2e-4, weight_decay=1e-4), CosineAnnealingLR (T_max=100, eta_min=1e-6), gradient clipping at 1.0. Modes: `--mode train` or `--mode enhance` (post-training inference with learned color correction).

```
python RD-DualNet2.py --mode train --low_dir <path> --high_dir <path> --epochs 100 --batch_size 16
python RD-DualNet2.py --mode enhance --eval_dir <path> --input_dir <path>
```

### Infrastructure

| File | Purpose |
|---|---|
| `config.py` | CUDA optimization (TF32, cuDNN benchmark, mixed precision hint), training config (gradient accumulation, num_workers), memory config (max_split_size_mb, GC threshold) |
| `install_darkvision.sh` | One-click env setup: detects Ubuntu version, CUDA version, creates venv, installs PyTorch with correct CUDA index, runs `test_environment.py` |
| `test_environment.py` | Environment validation: Python/PyTorch/CUDA/OpenCV/NumPy/Skimage versions, GPU tensor op test |
| `monitor_resources.py` | Real-time CPU/GPU monitoring daemon thread — psutil for CPU/RAM, gpustat for GPU utilization/memory/temperature |
| `image_enhance1.py` | Standalone inference script for DarkVisionNet — batch processes a folder, preserves original resolution |


2025 “Camera Academic Star” Imaging Algorithm Technology Competition

## Attempt.py

预估无监督学习算法无法完全生成合规图像，计划更换算法

## RD-DualNet.py

Retinex-DCP Guided Dual-Branch Network with Hybrid Supervision (RD-DualNet)

采用有监督学习，直接学习从暗图到 Ground Truth 的映射，这是获得高分的最可靠路径 。

通过融合 Retinex 和 DCP 先验，网络内部的注意力机制具有明确的物理意义，能更精准地分离和增强细节与光照。

使用 深度可分离卷积 和 通道注意力 ，在保证性能的同时大幅降低模型复杂度，符合移动端应用背景 

HybridLoss 结合了L1损失（细节保真）、感知损失（VGG特征对齐，提升视觉质量 ）和 GT-Mean Loss（整体亮度对齐 ），全方位逼近 Ground Truth。

使用 AdamW 优化器、梯度裁剪和余弦退火调度器，确保训练稳定高效。

此方案在算法逻辑上是创新的，并且完全针对赛题的评分标准（与Ground Truth对比）进行了优化。
