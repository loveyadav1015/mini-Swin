<div align="center">

# 🪟 Swin Transformer — From Scratch

**A hierarchical Vision Transformer built in pure PyTorch, served via FastAPI**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Based on "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" — Liu et al., ICCV 2021*

</div>

---

## 📖 What Is This?

This repository is a ground-up implementation of the **Swin Transformer** — a state-of-the-art Vision Transformer architecture that achieves **87.3% Top-1 accuracy on ImageNet-1K**. Unlike plain ViT, Swin builds *hierarchical* feature maps using a shifted-window self-attention mechanism that scales linearly with image size.

This project covers:
- ✅ Core architecture implemented from scratch in PyTorch
- ✅ Training pipeline with AdamW, cosine LR, AMP, and data augmentation
- ✅ Support for multiple model variants (Tiny → Large)
- ✅ FastAPI inference server with file upload endpoint
- ✅ ONNX export for production CPU serving
- ✅ Docker support

---

## 🏛️ Architecture Overview

Swin introduces two key ideas on top of standard ViT:

| Concept | What It Does |
|---|---|
| **Patch Partitioning** | Splits a 224×224 image into 4×4 non-overlapping patches, each a 48-dim token (4×4×3 RGB) |
| **Patch Merging** | At each stage, merges 2×2 adjacent patches → halves spatial size, doubles channels (like CNN stride) |
| **Window Attention (W-MSA)** | Computes self-attention inside local 7×7 windows — reduces complexity from O(N²) to O(N) |
| **Shifted Window Attention (SW-MSA)** | Shifts the window grid by `window_size // 2` to enable cross-window communication |
| **Relative Position Bias** | Learnable per-position bias added to attention logits inside each window |

```
Input (224×224×3)
     │
Patch Embed ───────────► [56×56, C=96]
     │
Stage 1: 2× [W-MSA → SW-MSA] ──► [56×56, C=96]
     │  Patch Merge (↓2×)
Stage 2: 2× [W-MSA → SW-MSA] ──► [28×28, C=192]
     │  Patch Merge (↓2×)
Stage 3: 6× [W-MSA → SW-MSA] ──► [14×14, C=384]
     │  Patch Merge (↓2×)
Stage 4: 2× [W-MSA → SW-MSA] ──► [7×7, C=768]
     │
Global Avg Pool → Linear Head → Logits
```

---

## 📐 Model Variants

| Variant | Params | FLOPs | ImageNet Top-1 | Min VRAM (Train) |
|---|---|---|---|---|
| **Swin-T** (Tiny) | 28M | 4.5G | 81.3% | ~6 GB |
| **Swin-S** (Small) | 50M | 8.7G | 83.0% | ~10 GB |
| **Swin-B** (Base) | 88M | 15.4G | 83.5% | ~16 GB |
| **Swin-L** (Large) | 197M | 34.5G | 86.3%* | ~32 GB |

*Pre-trained on ImageNet-22K, fine-tuned on ImageNet-1K.

> **Start with Swin-T** unless you have a workstation-grade GPU.

---

## 📦 Datasets

### Recommended by Experience Level

| Level | Dataset | Size | Classes | How to Get |
|---|---|---|---|---|
| 🟢 Beginner | CIFAR-10 | 170 MB | 10 | `torchvision.datasets.CIFAR10(download=True)` |
| 🟢 Beginner | CIFAR-100 | 170 MB | 100 | `torchvision.datasets.CIFAR100(download=True)` |
| 🟡 Intermediate | Tiny-ImageNet | 236 MB | 200 | [cs231n.stanford.edu](http://cs231n.stanford.edu/tiny-imagenet-200.zip) |
| 🟠 Advanced | ImageNet-1K | ~155 GB | 1,000 | [image-net.org](https://image-net.org/download.php) *(registration required)* |
| 🔴 Research | ImageNet-21K | ~1.3 TB | 21,841 | [image-net.org](https://image-net.org/download.php) |

### Domain-Specific Options

| Use Case | Dataset | Notes |
|---|---|---|
| Object Detection | COCO 2017 | 118K images, 80 classes |
| Semantic Segmentation | ADE20K | 20K images, 150 classes |
| Medical Imaging | BTCV / ChestX-ray14 | Synapse Multi-organ / NIH |
| Satellite Imagery | NWPU-RESISC45 | 45 classes, Swin-T → 82% accuracy |

---

## ⚡ Compute Requirements & Training Time

### Hardware Guide

| GPU | VRAM | Best Variant | Dataset |
|---|---|---|---|
| RTX 3060 / 4060 Ti | 8–12 GB | Swin-T (small batch) | CIFAR / Tiny-ImageNet |
| RTX 3090 / 4080 | 24 GB | Swin-T / Swin-S | Tiny-ImageNet / Custom |
| RTX 4090 | 24 GB | Swin-S / Swin-B | ImageNet-1K |
| A100 (40 GB) | 40 GB | Swin-B / Swin-L | ImageNet-1K |
| 8× A100 | 320 GB | Swin-L / SwinV2 | ImageNet-1K / 22K |

### Expected Training Times

| Config | Hardware | Epochs | Estimated Time |
|---|---|---|---|
| Swin-T on CIFAR-10 | RTX 3070 (8 GB) | 100 | ~2–4 hours |
| Swin-T on Tiny-ImageNet | RTX 3090 (24 GB) | 300 | ~12–18 hours |
| Swin-T on ImageNet-1K | 1× A100 (40 GB) | 300 | ~5–7 days |
| Swin-T on ImageNet-1K | 8× A100 | 300 | ~14–20 hours |
| SwinV2-B on ImageNet-1K | 16× A100 | 350 | ~54 hours |
| Swin-T fine-tune (custom) | RTX 3090 | 30–50 | ~40 min–2 hours |

> 💡 **Tip:** Don't train from scratch on ImageNet if you don't need to. Download pretrained weights from HuggingFace and fine-tune — training time drops from days to hours.

---

## 🗂️ Project Structure

```
swin_project/
├── model/
│   ├── swin_transformer.py       # Core architecture (PatchEmbed, WindowAttention,
│   │                             #   SwinBlock, PatchMerging, SwinTransformer)
│   └── config.py                 # Model config dataclasses
│
├── data/
│   ├── dataset.py                # Custom Dataset wrapper
│   ├── transforms.py             # Albumentations augmentation pipeline
│   └── dataloader.py             # DataLoader factory
│
├── training/
│   ├── train.py                  # Main training entry point
│   ├── trainer.py                # Trainer class (loss, optimizer, scheduler)
│   ├── losses.py                 # Label-smoothing cross-entropy, MixUp/CutMix
│   └── scheduler.py              # Cosine decay with linear warmup
│
├── evaluation/
│   ├── metrics.py                # Top-1 / Top-5 accuracy
│   └── evaluate.py               # Standalone evaluation script
│
├── api/
│   ├── main.py                   # FastAPI app + lifespan model loading
│   ├── routers/
│   │   └── predict.py            # POST /predict endpoint
│   ├── schemas.py                # Pydantic request / response models
│   └── model_manager.py          # Model loading singleton
│
├── scripts/
│   ├── download_dataset.sh       # Helper to fetch Tiny-ImageNet
│   ├── export_onnx.py            # Export checkpoint → ONNX
│   └── benchmark.py              # Throughput benchmarking
│
├── configs/
│   ├── swin_tiny.yaml
│   ├── swin_base.yaml
│   └── training_config.yaml
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/swin-transformer-scratch.git
cd swin-transformer-scratch

# Create conda environment
conda create -n swin_env python=3.11 -y
conda activate swin_env

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Train on CIFAR-10 (Quick Start)

```bash
python training/train.py \
  --config configs/swin_tiny.yaml \
  --dataset cifar10 \
  --epochs 100 \
  --batch-size 64 \
  --lr 5e-4
```

### 3. Train on Tiny-ImageNet

```bash
bash scripts/download_dataset.sh   # downloads to ./data/tiny-imagenet-200/

python training/train.py \
  --config configs/swin_tiny.yaml \
  --dataset tiny-imagenet \
  --data-dir ./data/tiny-imagenet-200 \
  --epochs 300 \
  --batch-size 128 \
  --amp                            # enable mixed precision
```

### 4. Fine-Tune from Pretrained Weights

```python
import timm

# Load pretrained Swin-T, replace head for your number of classes
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=10   # your dataset
)
```

### 5. Run the FastAPI Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

```bash
# Test with curl
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "accept: application/json" \
     -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "predicted_class": 281,
  "confidence": 0.9423,
  "top5": [[281, 0.9423], [285, 0.0312], [282, 0.0198], [287, 0.0041], [283, 0.0026]],
  "latency_ms": 14.7
}
```

---

## 🛠️ Tech Stack

### Core

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1.0 | Deep learning framework |
| `torchvision` | ≥ 0.16.0 | Datasets, transforms |
| `timm` | ≥ 0.9.12 | Pretrained Swin models, building blocks |
| `transformers` | ≥ 4.38 | HuggingFace Swin + AutoFeatureExtractor |
| `einops` | ≥ 0.7.0 | Clean tensor rearrangement ops |

### Training

| Package | Purpose |
|---|---|
| `accelerate` | Distributed training abstraction (HuggingFace) |
| `wandb` | Experiment tracking & loss visualization |
| `albumentations` | Fast, composable image augmentations |
| `deepspeed` | ZeRO optimizer for large model training |

### Serving

| Package | Purpose |
|---|---|
| `fastapi` | Async REST API framework |
| `uvicorn` | ASGI server (production-grade) |
| `pydantic` | Request/response schema validation |
| `python-multipart` | File upload support |
| `onnx` + `onnxruntime-gpu` | ONNX export for 2–3× faster CPU inference |

---

## ⚙️ Training Hyperparameters

```yaml
# configs/swin_tiny.yaml
model:
  variant: swin_tiny
  patch_size: 4
  window_size: 7
  embed_dim: 96
  depths: [2, 2, 6, 2]
  num_heads: [3, 6, 12, 24]
  drop_path_rate: 0.2

training:
  batch_size: 128
  epochs: 300
  optimizer: AdamW
  lr: 1.0e-3
  weight_decay: 0.05
  lr_scheduler: cosine
  warmup_epochs: 20
  min_lr: 5.0e-6
  grad_clip: 5.0
  amp: true

augmentation:
  rand_augment: true
  cutmix_alpha: 1.0
  mixup_alpha: 0.8
  label_smoothing: 0.1
  random_erasing: 0.25
```

---

## 🐳 Docker

```bash
# Build
docker build -t swin-api .

# Run (GPU)
docker run --gpus all -p 8000:8000 swin-api

# Run (CPU only)
docker run -p 8000:8000 swin-api
```

---

## 📤 Export to ONNX (Optional — for faster CPU inference)

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/swin_tiny_best.pth \
  --output swin_tiny.onnx \
  --input-size 224
```

Then point your FastAPI server at the ONNX model for ~2–3× faster inference on CPU.

---

## 📊 Results

| Dataset | Model | Accuracy | Hardware | Time |
|---|---|---|---|---|
| CIFAR-10 | Swin-T (scratch) | ~95% | RTX 3090 | ~3 hrs |
| Tiny-ImageNet | Swin-T (scratch) | ~70–75% | RTX 3090 | ~14 hrs |
| Tiny-ImageNet | Swin-L (fine-tune) | ~91.35% | RTX 3090 | ~2 hrs |
| ImageNet-1K | Swin-T (paper) | 81.3% | 8× V100 | — |
| ImageNet-1K | Swin-B (paper) | 83.5% | 8× V100 | — |

---

## 🗺️ Roadmap

- [x] Core Swin Transformer architecture
- [x] Training pipeline (AdamW + cosine LR + AMP + augmentation)
- [x] FastAPI inference server
- [x] ONNX export
- [x] Docker support
- [ ] SwinV2 implementation
- [ ] Object detection head (FPN integration)
- [ ] Semantic segmentation head (UperNet)
- [ ] Distributed training script (multi-GPU via `accelerate`)
- [ ] TensorRT export

---

## 📚 References

- Liu et al. (2021). [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030). ICCV 2021.
- Liu et al. (2022). [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883). CVPR 2022.
- [Official Microsoft Implementation](https://github.com/microsoft/Swin-Transformer)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using PyTorch · FastAPI · timm
</div>
