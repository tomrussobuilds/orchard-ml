← [Back to Main README](../../README.md)

# Docker Training Guide

## 🐳 Containerized Deployment

### Build Image

```bash
docker build -t visionforge:latest .
```

### Execution Modes

**Standard Mode** (Performance Optimized):
```bash
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18_adapted.yaml
```

**Strict Reproducibility Mode** (Bit-Perfect Determinism):
```bash
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -e IN_DOCKER=TRUE \
  -e DOCKER_REPRODUCIBILITY_MODE=TRUE \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -e PYTHONHASHSEED=42 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18_adapted.yaml
```

> [!NOTE]
> - `TORCH_HOME` and `MPLCONFIGDIR` prevent permission errors in containerized environments
> - `CUBLAS_WORKSPACE_CONFIG` is required for CUDA determinism
> - `--gpus all` requires NVIDIA Container Toolkit

---
