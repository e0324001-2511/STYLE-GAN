# StyleGAN-Style Generator on CIFAR-10 (TensorFlow / Keras)

A compact, educational implementation of a **StyleGAN-style GAN** trained on a subset of CIFAR-10. The notebook builds a mapping network, a style-modulated generator (AdaIN + per-layer noise), and a small convolutional discriminator, then demonstrates **style mixing** and **latent interpolation**.

It is designed to be small and quick enough for a classroom or laptop run, not to reproduce the full NVIDIA StyleGAN.

---

## Contents

| File | Description |
| --- | --- |
| `StyleGAN.ipynb` | Full pipeline: data loading, model definition, training, and visualizations |

---

## Features

- **Mapping network:** 8 fully connected layers that map latent `z` (128-d) to intermediate latent `w` (128-d), with input normalization and LeakyReLU.
- **Style-based generator:**
  - Learned 4×4 constant input instead of feeding `z` directly
  - **AdaIN** layers (instance-style normalization plus learned per-channel scale and bias from `w`)
  - Learned **per-layer noise injection**
  - Progressive resolution through 4×4 → 8×8 → 16×16 → 32×32 with nearest-neighbor upsampling
  - `tanh` RGB output
- **Discriminator:** 4 strided convolution blocks (32→64→128→128 channels) followed by a dense logit head.
- **Loss:** Non-saturating logistic GAN loss (softplus formulation).
- **Optimizer:** Adam, `lr=2e-4`, `beta_1=0.0`, `beta_2=0.99` for both networks.
- **Demos:**
  - Loss curves and a grid of generated samples from a fixed latent batch
  - **Style mixing:** combine early styles from image A with later styles from image B at layers 2, 4, and 6
  - **Latent interpolation:** 9-step linear interpolation between two latent vectors

---

## Requirements

- Python 3.9+
- TensorFlow 2.x (Keras 3 style API is used, e.g. `LeakyReLU(negative_slope=...)`, so TensorFlow 2.16+ is recommended)
- NumPy
- Matplotlib
- Jupyter Notebook / JupyterLab (or Google Colab)

```bash
pip install tensorflow numpy matplotlib jupyter
```

A GPU is optional but speeds up training considerably. CIFAR-10 is downloaded automatically by Keras on first run.

---

## Usage

```bash
git clone <your-repo-url>
cd <your-repo>
jupyter notebook StyleGAN.ipynb
```

Run the cells in order (**Kernel → Restart & Run All**). The notebook is organized as follows:

1. **Setup:** imports, seeding (`SEED = 42`), device check
2. **Data:** loads CIFAR-10, takes the first 20,000 training images, scales to `[-1, 1]`, batches (64)
3. **Models:** defines `MappingNetwork`, `AdaIN`, `StyledConv`, `StyleGenerator`, `Discriminator` and prints parameter counts
4. **Training:** 10 epochs with alternating discriminator / generator updates
5. **Results:** loss plot and generated sample grid
6. **Style mixing:** crossover of two latent styles at different layers
7. **Latent interpolation:** smooth transition between two samples

---

## Configuration

Key hyperparameters (edit at the top of the relevant cells):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `TRAIN_LIMIT` | `20000` | Number of CIFAR-10 images used |
| `BATCH` | `64` | Batch size |
| `Z_DIM` / `W_DIM` | `128` | Latent and intermediate latent dimensions |
| `CHANNELS` | `{4:128, 8:128, 16:64, 32:32}` | Feature channels per resolution |
| `EPOCHS` | `10` | Training epochs |
| `SEED` | `42` | Random seed for reproducibility |

For better image quality, try increasing `TRAIN_LIMIT` to the full 50,000 images and `EPOCHS` to 50 or more.

---

## Expected Output

- Console logs of the discriminator and generator loss per epoch with elapsed time
- A training-loss plot
- An 8×8 grid of generated 32×32 images
- A style-mixing comparison figure (Source A, Source B, and three mixes)
- A latent interpolation strip

With only 10 epochs on 20k images, samples will be blurry and abstract. This is expected for a short educational run.

---

## Differences from the Official StyleGAN

This is a simplified version intended for learning. It does **not** include:

- Progressive growing
- Weight demodulation / modulated convolutions (StyleGAN2)
- Equalized learning rate
- Path-length or R1 regularization
- Truncation trick, EMA of generator weights, or mixing regularization during training

---

## References

- Karras, Laine, Aila. *A Style-Based Generator Architecture for Generative Adversarial Networks* (StyleGAN), CVPR 2019.
- Goodfellow et al. *Generative Adversarial Nets*, NeurIPS 2014.
- Krizhevsky. *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10), 2009.

---

## License

Add your preferred license here (e.g. MIT).
