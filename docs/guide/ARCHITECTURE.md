← [Back to Main README](../../README.md)

# Supported Models

## Pretrained Weights and Transfer Learning

All models except MiniCNN are initialized with **pretrained weights** — parameters learned by training on ImageNet, a large-scale dataset of natural images. Instead of starting from random values, the network begins with convolutional filters that already encode useful visual features: edge detectors, texture patterns, color gradients, and shape representations.

**Transfer learning** leverages this prior knowledge: the pretrained feature extractor is kept (or fine-tuned), and only the final classifier layer is replaced to match the target task (e.g., 9 disease classes instead of 1000 ImageNet categories). This dramatically reduces the amount of labeled data and training time needed to reach strong performance, which is especially valuable in domains like medical imaging where annotated samples are scarce.

### ImageNet Variants

| Dataset | Images | Classes | Used by |
|---------|--------|---------|---------|
| **ImageNet-1k** | ~1.2M | 1,000 | ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-Tiny (baseline) |
| **ImageNet-21k** | ~14M | 21,841 | ViT-Tiny (augreg variants) |

ImageNet-21k provides a broader visual vocabulary at the cost of noisier labels. ViT-Tiny benefits from this larger pretraining because transformers are more data-hungry than CNNs. The `augreg_in21k_ft_in1k` variant combines the best of both: pretrained on 21k, then fine-tuned on the cleaner 1k labels.

> [!IMPORTANT]
> **Data leakage**: when using ImageNet-pretrained weights, ensure your target dataset has no overlap with ImageNet. All datasets currently supported by VisionForge (MedMNIST, Galaxy10) come from entirely different domains (medical imaging, astronomy) and share zero samples with ImageNet. If you add a custom dataset of natural images, verify that it does not contain ImageNet samples — otherwise evaluation metrics will be inflated because the model has already seen those images during pretraining.

### Weight Morphing

Pretrained weights assume RGB input (3 channels) at a specific resolution. When the target domain differs — grayscale medical images (1 channel) or lower resolution (28x28 instead of 224x224) — the weights must be **adapted** rather than discarded:

- **Channel averaging**: compresses 3-channel filters into 1-channel by averaging across the RGB dimension, preserving the learned spatial patterns
- **Spatial interpolation** (ResNet-18 28x28 only): resizes 7x7 kernel weights to 3x3 via bicubic interpolation to match the smaller stem

The exact transformations and tensor dimensions are documented under each model below.

---

## ResNet-18 (Multi-Resolution: 28x28 / 224x224)

Adaptive ResNet-18 that automatically selects the appropriate stem configuration based on `cfg.dataset.resolution`.

### 28x28 Mode (Low-Resolution)

Standard ResNet-18 is optimized for 224x224 ImageNet inputs. Direct application to 28x28 domains causes catastrophic information loss. The 28x28 mode performs architectural surgery on the ResNet-18 stem:

| Layer | Standard ResNet-18 | VisionForge 28x28 Mode | Rationale |
|-------|-------------------|----------------------|-----------|
| **Input Conv** | 7x7, stride=2, pad=3 | **3x3, stride=1, pad=1** | Preserve spatial resolution |
| **Max Pooling** | 3x3, stride=2 | **Identity (bypassed)** | Prevent 75% feature loss |
| **Stage 1 Input** | 56x56 (from 224) | **28x28 (from 28)** | Native resolution entry |

**Weight Transfer (28x28):**

Pretrained 7x7 ImageNet weights are spatially interpolated to the smaller 3x3 kernel via bicubic interpolation:

```math
W_{\text{3x3}} = \mathcal{I}_{\text{bicubic}}(W_{\text{7x7}}, \text{size}=(3, 3)) \quad \text{where} \quad W_{\text{7x7}} \in \mathbb{R}^{64 \times 3 \times 7 \times 7}
```

For grayscale inputs, channel averaging is applied before interpolation:

```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W[:, c, :, :]
```

This two-step process (channel compress + spatial resize) preserves learned edge detectors while adapting to both single-channel input and smaller kernel geometry.

### 224x224 Mode (High-Resolution)

At 224x224, ResNet-18 uses the standard architecture with no structural modifications:

| Layer | Specification | Notes |
|-------|--------------|-------|
| **Input Conv** | 7x7, stride=2, pad=3 | Standard ImageNet configuration |
| **Max Pooling** | 3x3, stride=2 | Full downsampling pipeline |

**Weight Transfer (224x224):**

No spatial interpolation is needed. For grayscale inputs, the pretrained RGB weights are compressed via channel averaging:

```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W[:, c, :, :] \quad \text{where} \quad W \in \mathbb{R}^{64 \times 3 \times 7 \times 7}
```

---

## MiniCNN (28x28)

A compact, custom architecture designed specifically for low-resolution medical imaging. No pretrained weights — trained from scratch.

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Architecture** | 3 conv blocks + global pooling | Fast convergence with minimal parameters |
| **Parameters** | ~94K | 220x fewer than ResNet-18 |
| **Input Processing** | 28x28 → 14x14 → 7x7 → 1x1 | Progressive spatial compression |
| **Regularization** | Configurable dropout before FC | Overfitting prevention |

**Advantages:**
- **Speed**: 2-3 minutes for full 60-epoch training on GPU
- **Efficiency**: Ideal for rapid prototyping and ablation studies
- **Interpretability**: Simple architecture for educational purposes

---

## EfficientNet-B0 (224x224)

Implements compound scaling (depth, width, resolution) for optimal parameter efficiency.

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | Mobile Inverted Bottleneck Convolution (MBConv) | Memory-efficient feature extraction |
| **Parameters** | ~4.0M | 50% fewer than ResNet-50 |
| **Pretrained Weights** | ImageNet-1k | Strong initialization for transfer learning |

**Weight Transfer:**

The first convolutional layer (`features[0][0]`) is a Conv2d(3, 32, 3x3). For grayscale inputs, pretrained RGB weights are compressed via channel averaging:

```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W[:, c, :, :] \quad \text{where} \quad W \in \mathbb{R}^{32 \times 3 \times 3 \times 3}
```

---

## Vision Transformer Tiny (ViT-Tiny) (224x224)

Patch-based attention architecture with multiple pretrained weight variants.

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | 12-layer transformer encoder | Global context modeling via self-attention |
| **Parameters** | ~5.5M | Comparable to EfficientNet-B0 |
| **Patch Size** | 16x16 (196 patches from 224x224) | Efficient sequence length for transformers |

**Supported Weight Variants:**
1. `vit_tiny_patch16_224.augreg_in21k_ft_in1k`: ImageNet-21k pretrained, fine-tuned on 1k (recommended)
2. `vit_tiny_patch16_224.augreg_in21k`: ImageNet-21k pretrained (requires custom head tuning)
3. `vit_tiny_patch16_224`: ImageNet-1k baseline

**Weight Transfer:**

The patch embedding layer projects 16x16 patches into 192-dimensional tokens. For grayscale inputs, the 3-channel projection weights are averaged:

```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W[:, c, :, :] \quad \text{where} \quad W \in \mathbb{R}^{192 \times 3 \times 16 \times 16}
```

---

## ConvNeXt-Tiny (224x224)

Modern ConvNet architecture incorporating design principles from Vision Transformers.

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | Inverted bottlenecks + depthwise convolutions | Improved efficiency and accuracy |
| **Parameters** | ~27.8M | Competitive with transformers |
| **Pretrained Weights** | ImageNet-1k | Strong initialization for transfer learning |
| **Stem** | 4x4 conv, stride 4 (patchification) | Efficient spatial downsampling |

**Key Design Choices:**
- Depthwise convolutions with larger kernels (7x7)
- Layer normalization instead of batch normalization
- GELU activation functions
- Fewer activation and normalization layers than traditional CNNs

**Weight Transfer:**

The stem layer (`features[0][0]`) is a Conv2d(3, 96, 4x4, stride=4). For grayscale inputs, pretrained RGB weights are compressed via channel averaging:

```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W[:, c, :, :] \quad \text{where} \quad W \in \mathbb{R}^{96 \times 3 \times 4 \times 4}
```

---
