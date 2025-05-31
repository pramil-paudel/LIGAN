# LiGAN: Lensless Image GAN for Secure Steganography

**LiGAN** is a deep learning framework that uses lensless imaging principles to hide and recover secret images using a GAN-based architecture. The system is designed for high-quality steganography, where a **lensless image** is hidden within a normal cover image using a **Swin Transformer**-based network, and then **revealed** using a custom GAN decoder that reconstructs the hidden message in either low or high resolution.

---

## Key Features

- **Lensless Secret Embedding**  
  Leverages lensless imaging technology to securely embed secret images into natural-looking containers.

- **SwinT-based Hiding Network**  
  Uses a Swin Transformer encoder to hide the lensless secret image into the cover image without noticeable distortion.

- **🛠️ Custom GAN Reveal Network**  
  Recovers the secret message from the container using a GAN:
  - Outputs **low-resolution reconstructions**, or
  - Upscales to **4× resolution** to match the container image

- **📊 High-Fidelity Evaluation**  
  Supports PSNR, SSIM, and perceptual loss metrics for both container quality and secret recovery fidelity.

---

##Project Structure

- `gan/` – GAN components including generator, discriminator, and loss functions  
- `lensless/` – Lensless imaging utilities and data preprocessing modules  
- `stego/` – Steganography-specific hiding and revealing networks  
- `runs/` – Ready-to-run experiment setups:
  - `low_res/` – Scripts and configs for low-resolution secret recovery
  - `high_res/` – Scripts and configs for high-resolution (4× upscaled) recovery  

---

## How It Works

### Hiding Phase
A Swin Transformer takes a `cover image` and a `lensless secret image`, and generates a `container image` that is visually indistinguishable from the cover but embeds the secret.

### Reveal Phase
A GAN-based network extracts the secret from the container and reconstructs:
- Either a **low-resolution** version of the original secret
- Or a **super-resolved** version matching the container's size (typically 4× the input)

---
![lensless_stegnanography.PNG](images%2Flensless_stegnanography.PNG)

## Metrics Used

- **PSNR** – Peak Signal-to-Noise Ratio
- **SSIM** – Structural Similarity Index
- **Perceptual Loss** – Based on VGG features

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.11
- Additional packages: `timm`, `einops`, `pytorch-msssim`, `matplotlib`, `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
