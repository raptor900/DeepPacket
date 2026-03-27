# Deep Packet

PyTorch implementation of the approach described in:

> Lotfollahi, M., Jafari Siavoshani, M., Shirali Hossein Zade, R., & Saberian, M. (2017).
> *Deep Packet: A Novel Approach For Encrypted Traffic Classification Using Deep Learning.*
> [arXiv:1709.02656](https://arxiv.org/abs/1709.02656)

## Overview

Deep Packet uses deep learning to classify encrypted network traffic (e.g., VPN/non-VPN, application type) **without decryption**. It operates on raw packet bytes, treating traffic classification as an image/signal classification problem.

Two architectures are implemented:

### 1. DeepPacketCNN (1D CNN)

Direct end-to-end classification from raw bytes:

```
Input (1 × 1500 bytes)
    ↓
Conv1D (200 filters, kernel=5, stride=2) → BN → ReLU
    ↓
Conv1D (100 filters, kernel=4, stride=1) → BN → ReLU
    ↓
AvgPool (2)
    ↓
FC layers: 37200 → 600 → 500 → 400 → 300 → 200 → 100 → 50
    ↓
Output (n_classes) → LogSoftmax
```

### 2. StackedAutoEncoder

Pre-trained stacked autoencoder with classification head:

```
Input (1500)
    ↓
AE1: 1500 → 400  (greedy pre-training)
AE2: 400  → 300
AE3: 300  → 200
AE4: 200  → 100
AE5: 100  → 50
    ↓
FC: 50 → n_classes → LogSoftmax
```

Each autoencoder is pre-trained greedily (layer-by-layer) using reconstruction loss, then the full network is fine-tuned end-to-end.

## Usage

```python
import torch
from models import DeepPacketCNN, StackedAutoEncoder

# CNN model
model = DeepPacketCNN(n_classes=10)
x = torch.randn(32, 1, 1500)  # batch of 32 packets
log_probs = model(x)  # (32, 10)

# Stacked AutoEncoder model
sae = StackedAutoEncoder(n_classes=10)
x_flat = torch.randn(32, 1500)  # flattened packets
log_probs = sae(x_flat)  # (32, 10)
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9
