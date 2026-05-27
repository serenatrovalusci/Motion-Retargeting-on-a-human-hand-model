# Hand Motion Reconstruction from Minimal Inputs through Latent-Space Learning

A real-time, data-driven framework for hand motion retargeting that reconstructs full joint configurations of a virtual hand from only **four high-level inputs** acquired from a Weart TouchDIVER G1 haptic glove. The approach yields anatomically consistent poses with low latency in Unity, despite minimal sensing.

> **Authors:** Charlotte Ludovica Primiceri · Serena Trovalusci · Diana Ioana Bubenek Turconi · Giordano Pagano  
> **Supervisors:** Emanuele De Santis · Marilena Vendittelli  
> **Institution:** Dipartimento di Ingegneria Informatica, Automatica e Gestionale (DIAG), Sapienza University of Rome  
> **Report:** [`Hand_Motion_Reconstruction.pdf`](Hand_Motion_Reconstruction_from_Minimal_Inputs_through_Latent_Space_Learning.pdf)

---

## What is this project?

Existing hand retargeting pipelines rely on dense sensing (e.g., full keypoint tracking) or optimization-heavy inverse kinematics, which limits portability and latency in interactive settings. This project instead pursues **minimal sensing** — just four scalars — with a learning-based model that restores a full, biomechanically consistent 45-DoF hand pose in real time.

```
┌──────────────────────────────────────────────────────────────────┐
│                         Pipeline Overview                        │
│                                                                  │
│   Weart TouchDIVER G1                                            │
│   [ThumbClosure, IndexClosure, MiddleClosure, ThumbAbduction]    │
│                    │  4 scalars                                  │
│                    ▼                                             │
│             Neural Network                                       │
│         (FCNN or Transformer)                                    │
│                    │  62 outputs (sin/cos encoded)               │
│                    ▼                                             │
│          45 joint angles  ──►  Unity virtual hand (20 Hz)        │
└──────────────────────────────────────────────────────────────────┘
```

Human hand movements exhibit **synergies** — joints move in coordinated, correlated patterns rather than independently. This project exploits that property through dimensionality reduction (PCA and Autoencoders) to reconstruct realistic hand poses from a limited input set.

---

## Models

| Model | Architecture | Training Speed | Inference | Generalization |
|-------|-------------|---------------|-----------|----------------|
| **FCNN** | `Input(4) → 512 → 256 → 128 → 64 → Output(62)` with LeakyReLU + BatchNorm/Dropout | Very fast | Extremely fast | Limited (linear/simple cases) |
| **Transformer** | Input Embedding → Positional Encoding → 3× Encoder Layers (MHA 128-dim, 4 heads) → 5 per-finger Output Heads | Slower | Fast (real-time viable) | Strong (nonlinear/noisy data) |

Both models output **62 values**: 45 joint angles, with problematic joints (all thumb joints, Middle03, Index02/03) encoded as sine/cosine pairs to avoid Euler angle discontinuities at ±180°.

---

## Dimensionality Reduction Strategies

Three approaches were explored to enforce biomechanical plausibility:

| Strategy | Description | Outcome |
|----------|-------------|---------|
| **Direct PCA Output** | Models predict PCA coefficients directly; inverse PCA to recover joint angles | Good compression, acceptable accuracy down to 15 components |
| **PCA in the Loss** | Models predict full 62-D output; loss computed in PCA space | Low test loss but erratic individual joint behavior |
| **Autoencoder Latent Loss** *(best)* | Frozen encoder maps predictions and ground truth to latent space; MSE in latent space | Best visual quality and joint-level accuracy |

The autoencoder architecture is symmetric: `62 → 512 → 256 → L → 256 → 512 → 62` with ReLU activations, trained with latent dimensions `L ∈ {10, 15, 30, 45}`. The encoder is frozen during predictor training; only the regression model is optimized.

---

## Repository Structure

```
.
├── train.py                          # FCNN / Transformer training (baseline)
├── train_losspca.py                  # Training with PCA-based loss
├── train_lossencoder.py              # Training with autoencoder latent loss
├── autoencoder.py                    # Autoencoder definition and training
├── models/                           # FCNN and Transformer model definitions
├── HandDataLogger.cs                 # Unity C# script for dataset recording (20 Hz)
├── inference_server.py               # Python TCP server for real-time Unity inference
├── dataset/                          # Recorded CSV files (~17,000 samples)
├── video_results/                    # Demo videos per experiment
│   ├── training_losspca_results/
│   └── training_lossencoder_results/
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/serenatrovalusci/Hand-Motion-Reconstruction-from-Minimal-Inputs-through-Latent-Space-Learning.git
cd Hand-Motion-Reconstruction-from-Minimal-Inputs-through-Latent-Space-Learning
```

### 2. Install dependencies

```bash
pip install torch numpy pandas scikit-learn
```

### 3. Train a model

```bash
# Baseline (no dimensionality reduction)
python train.py

# With PCA-based loss
python train_losspca.py

# With autoencoder latent loss (best results)
python train_lossencoder.py
```

### 4. Run real-time inference

Start the Python server, then play the Unity scene:

```bash
python inference_server.py
```

> Requires Unity with the Weart SDK and the `HandDataLogger.cs` / inference scripts integrated into the project scene.

---

## Results

### Best Configuration: Transformer + Autoencoder Latent Loss (L = 15)

The Transformer trained with a 15-dimensional autoencoder latent loss achieves the best balance between accuracy, visual quality, and computational efficiency — delivering natural, biomechanically consistent hand motion in real time.

| Latent Dimension | FCNN Test MSE | Transformer Test MSE |
|-----------------|--------------|---------------------|
| 10 | 0.0062 | 0.0053 |
| **15** | 0.0061 | **0.0051** |
| 30 | 0.0043 | 0.0036 |
| 45 | 0.0049 | 0.0040 |

Key findings:

- The **Transformer consistently outperforms the FCNN** across all latent dimensions, especially for joints with higher variability.
- The **15-dimensional latent space** strikes the best balance: visually consistent motion with significantly reduced training complexity.
- Adding a **constraint loss term** (ReLU-based penalty for out-of-range joint predictions) further stabilized outputs and improved validation loss.
- Our baseline achieves a **per-joint MAE of ~2.5°**, compared to ~15° reported by Glauser et al. [7] on a similar glove-based task.
- The **PCA-in-loss** approach yielded low test loss but erratic individual joints: small errors in PCA space cause large deviations after inverse PCA, particularly for non-dominant joints.

---

## Demo — Best Performance

**Transformer + Autoencoder Latent Loss (L = 15)** — real-time hand reconstruction in Unity.

https://github.com/user-attachments/assets/a3feec20-c0c4-4050-8615-d4f05fa425de

---

## Acknowledgements

This project was developed as original work at the DIAG department of Sapienza University of Rome, under the supervision of Emanuele De Santis and Marilena Vendittelli.
