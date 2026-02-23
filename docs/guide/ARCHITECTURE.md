← [Back to Home](../index.md)

<h1 align="center">Supported Models</h1>

<h2>Pretrained Weights and Transfer Learning</h2>

All models except MiniCNN are initialized with **pretrained weights** — parameters learned by training on ImageNet, a large-scale dataset of natural images. Instead of starting from random values, the network begins with convolutional filters that already encode useful visual features: edge detectors, texture patterns, color gradients, and shape representations.

**Transfer learning** leverages this prior knowledge: the pretrained feature extractor is kept (or fine-tuned), and only the final classifier layer is replaced to match the target task (e.g., 9 disease classes instead of 1000 ImageNet categories). This dramatically reduces the amount of labeled data and training time needed to reach strong performance, which is especially valuable in domains like medical imaging where annotated samples are scarce.

<h3>ImageNet Variants</h3>

| Dataset | Images | Classes | Used by |
|---------|--------|---------|---------|
| **ImageNet-1k** | ~1.2M | 1,000 | ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-Tiny (baseline) |
| **ImageNet-21k** | ~14M | 21,841 | ViT-Tiny (augreg variants) |

ImageNet-21k provides a broader visual vocabulary at the cost of noisier labels. ViT-Tiny benefits from this larger pretraining because transformers are more data-hungry than CNNs. The `augreg_in21k_ft_in1k` variant combines the best of both: pretrained on 21k, then fine-tuned on the cleaner 1k labels.

> [!IMPORTANT]
> **Data leakage**: when using ImageNet-pretrained weights, ensure your target dataset has no overlap with ImageNet. MedMNIST and Galaxy10 come from entirely different domains (medical imaging, astronomy) and share zero samples with ImageNet. CIFAR-10/100 shares semantic categories with ImageNet (cat, dog, bird, etc.) but uses distinct 32×32 images from a separate collection — pretrained features may provide an advantage that would not generalize to truly novel domains. If you add a custom dataset of natural images, verify that it does not contain ImageNet samples — otherwise evaluation metrics will be inflated because the model has already seen those images during pretraining.

<h3>Extended Weight Sources via timm</h3>

The five built-in architectures default to torchvision ImageNet-1k weights. For broader experimentation, any model in the [timm](https://huggingface.co/docs/timm) registry can be used via the `timm/` prefix, unlocking hundreds of pretrained weight variants from different sources:

| Source | Example | Overlap with ImageNet |
|--------|---------|----------------------|
| **ImageNet-1k** (supervised) | `timm/resnet18.a1_in1k` | Full (labels + images) |
| **ImageNet-21k** (supervised) | `timm/convnext_tiny.fb_in22k` | Partial (superset labels) |
| **SSL on YFCC100M** | `timm/resnet50.fb_ssl_yfcc100m_ft_in1k` | Low (SSL pretraining, then IN1k fine-tune) |
| **SWSL on Instagram 1B** | `timm/resnet50.fb_swsl_ig1b_ft_in1k` | Low (semi-weakly supervised on IG, then IN1k fine-tune) |
| **CLIP** | `timm/vit_base_patch16_clip_224.openai` | None (contrastive text-image pretraining) |
| **DINOv2** | `timm/vit_small_patch14_dinov2.lvd142m` | None (self-supervised, no labels) |
| **No pretrained** | Any model with `pretrained: false` | None |

**When to choose what:**
- **MedMNIST / Galaxy10** (no ImageNet overlap): ImageNet-1k pretrained is safe and recommended
- **CIFAR-10/100** (semantic overlap): Consider `pretrained: false` or SSL/CLIP weights for fairer benchmarks
- **Custom natural-image datasets**: Evaluate overlap risk before using ImageNet weights

Use `model_pool` in Optuna recipes to compare architectures and weight sources automatically — see the CIFAR optimization recipes for examples.

<h3>Weight Morphing</h3>

Pretrained weights assume RGB input (3 channels) at 224x224 resolution. When the target domain differs — grayscale medical images (1 channel) or lower resolution (28x28, 64x64) — the weights must be **adapted** rather than discarded:

- **Channel averaging**: compresses 3-channel filters into 1-channel by averaging across the RGB dimension, preserving the learned spatial patterns
- **Spatial interpolation** (ResNet-18 ≤32×32 only): resizes 7x7 kernel weights to 3x3 via bicubic interpolation to match the smaller stem

The exact transformations and tensor dimensions are documented under each model below.

---

<h2>ResNet-18 (Multi-Resolution: 28x28 / 32x32 / 64x64 / 224x224)</h2>

Adaptive ResNet-18 that automatically selects the appropriate stem configuration based on `cfg.dataset.resolution`.

<h3>≤32×32 Mode (Low-Resolution: 28x28, 32x32)</h3>

Standard ResNet-18 is optimized for 224x224 ImageNet inputs. Direct application to ≤32×32 domains causes catastrophic information loss. This mode performs architectural surgery on the ResNet-18 stem:

| Layer | Standard ResNet-18 | Orchard ML ≤32×32 Mode | Rationale |
|-------|-------------------|----------------------|-----------|
| **Input Conv** | 7x7, stride=2, pad=3 | **3x3, stride=1, pad=1** | Preserve spatial resolution |
| **Max Pooling** | 3x3, stride=2 | **Identity (bypassed)** | Prevent 75% feature loss |
| **Stage 1 Input** | 56x56 (from 224) | **28x28 or 32x32 (native)** | Native resolution entry |

**Weight Transfer (≤32×32):**

Pretrained 7x7 ImageNet weights are spatially interpolated to the smaller 3x3 kernel via bicubic interpolation:

```math
W_{\text{3x3}} = \mathcal{I}_{\text{bicubic}}(W_{\text{7x7}}, \text{size}=(3, 3)) \quad \text{where} \quad W_{\text{7x7}} \in \mathbb{R}^{64 \times 3 \times 7 \times 7}
```

For grayscale inputs, channel averaging is applied before interpolation:

```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W[:, c, :, :]
```

This two-step process (channel compress + spatial resize) preserves learned edge detectors while adapting to both single-channel input and smaller kernel geometry.

<h3>64x64 / 224x224 Mode (Standard Stem)</h3>

At 64x64 and 224x224, ResNet-18 uses the standard architecture with no structural modifications. The standard stem (7x7 conv stride-2 + MaxPool) produces valid spatial maps at both resolutions:

| Resolution | Spatial progression | Final feature map |
|-----------|-------------------|------------------|
| **64x64** | 64→32→16→8→4→2→1 | 1x1 (via AdaptiveAvgPool) |
| **224x224** | 224→112→56→28→14→7→1 | 1x1 (via AdaptiveAvgPool) |

| Layer | Specification | Notes |
|-------|--------------|-------|
| **Input Conv** | 7x7, stride=2, pad=3 | Standard ImageNet configuration |
| **Max Pooling** | 3x3, stride=2 | Full downsampling pipeline |

**Weight Transfer (64x64 / 224x224):**

No spatial interpolation is needed. For grayscale inputs, the pretrained RGB weights are compressed via channel averaging:

```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W[:, c, :, :] \quad \text{where} \quad W \in \mathbb{R}^{64 \times 3 \times 7 \times 7}
```

---

<h2>MiniCNN (28x28 / 32x32 / 64x64)</h2>

A compact, custom architecture designed for low-resolution imaging. No pretrained weights — trained from scratch. Uses `AdaptiveAvgPool2d((1,1))` so it works at any spatial resolution.

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Architecture** | 3 conv blocks + global pooling | Fast convergence with minimal parameters |
| **Parameters** | ~95K | 220x fewer than ResNet-18 |
| **Input Processing** | Adaptive (e.g., 28→14→7→1 or 64→32→16→1) | Progressive spatial compression |
| **Regularization** | Configurable dropout before FC | Overfitting prevention |

**Advantages:**
- **Speed**: 2-3 minutes for full 60-epoch training on GPU
- **Efficiency**: Ideal for rapid prototyping and ablation studies
- **Interpretability**: Simple architecture for educational purposes

---

<h2>EfficientNet-B0 (224x224)</h2>

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

<h2>Vision Transformer Tiny (ViT-Tiny) (224x224)</h2>

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

<h2>ConvNeXt-Tiny (224x224)</h2>

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
