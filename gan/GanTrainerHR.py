import os
import pickle
import random
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_msssim
import matplotlib.pyplot as plt
import gan.GanUtils as utils
import math


# === SSIM Loss ===
def ssim_loss(pred, target):
    return 1 - pytorch_msssim.ssim(pred.clamp(0, 1), target.clamp(0, 1), data_range=1.0)


# === VGG Perceptual Loss ===
class VGGPerceptualLoss(nn.Module):
    def __init__(self, epoch, layers_phase1=(1, 3), layers_phase2=(1,), layer_weights_phase1=None,
                 layer_weights_phase2=None, weight=1.0):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg.children())[:max(layers_phase1 + layers_phase2) + 1])
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.epoch = epoch
        self.layers_phase1 = layers_phase1
        self.layers_phase2 = layers_phase2
        self.layer_weights_phase1 = layer_weights_phase1 or {1: 1.0, 3: 0.8}
        self.layer_weights_phase2 = layer_weights_phase2 or {1: 1.0}
        self.weight = weight

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        x = (x - 0.5) * 2
        y = (y - 0.5) * 2

        loss = 0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if self.epoch < 50 and i in self.layers_phase1:
                loss += self.layer_weights_phase1.get(i, 1.0) * F.l1_loss(x, y)
            elif self.epoch >= 50 and i in self.layers_phase2:
                loss += self.layer_weights_phase2.get(i, 1.0) * F.l1_loss(x, y)

        return loss


# ==
# = Stochastic Lambda Schedule with Decay and Noise ===
# lambda_mae, lambda_percep, lambda_ssim, lambda_adv
def get_lambda_schedule(epoch):
    # Base weights per phase
    if epoch < 5:
        base = [10.0, 0.0, 1.0, 0.0]  # Strong MAE only
    elif epoch < 15:
        base = [5.0, 0.1, 0.8, 0.1]
    elif epoch < 30:
        base = [2.0, 0.3, 0.5, 0.4]
    elif epoch < 80:
        base = [1.5, 0.0, 0.3, 0.1]
    else:
        adv_jitter = 0.03 * (1 + math.sin(epoch))  # Smooth ~[0.0, 0.06]
        base = [2.0, 0.8, 0.2, adv_jitter]
    # Add small Gaussian noise to simulate stochastic adaptation
    noise_scale = 0.05  # ±5% noise
    noisy = [max(0.0, w + random.uniform(-noise_scale, noise_scale) * w) for w in base]
    return tuple(noisy)


# === Logging and Plotting ===
def save_losslog_pkl(log_path, data):
    with open(log_path, 'wb') as f:
        pickle.dump(data, f)


def load_losslog_pkl(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'rb') as f:
            return pickle.load(f)
    return []


def plot_loss_contributions(log_data, save_path):
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt

    epoch_dict = defaultdict(list)
    for entry in log_data:
        epoch_dict[entry['epoch']].append(entry['contrib'])

    epochs = sorted(epoch_dict.keys())
    avg_contrib = np.array([np.mean(epoch_dict[e], axis=0) for e in epochs])

    plt.figure(figsize=(6, 6))
    styles = ['-', '--', '-.', (0, (3, 1, 1, 1))]
    labels = ["MAE", "Percep", "SSIM", "Adv"]
    for i in range(4):
        plt.plot(epochs, avg_contrib[:, i], linestyle=styles[i], color='black', label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Average Weighted Loss")
    plt.title("Loss Contributions per Epoch")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_values(log_data, save_path):
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt

    epoch_dict = defaultdict(list)
    for entry in log_data:
        epoch_dict[entry['epoch']].append(entry['losses'])

    epochs = sorted(epoch_dict.keys())
    avg_losses = np.array([np.mean(epoch_dict[e], axis=0) for e in epochs])

    plt.figure(figsize=(6, 6))
    styles = ['-', '--', '-.', (0, (3, 1, 1, 1))]
    labels = ["MAE", "Percep", "SSIM", "Adv"]
    for i in range(4):
        plt.plot(epochs, avg_losses[:, i], linestyle=styles[i], color='black', label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Average Raw Loss Value")
    plt.title("Loss Values per Epoch")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_percentage(log_data, save_path):
    from collections import defaultdict
    epoch_dict = defaultdict(list)
    for entry in log_data:
        epoch_dict[entry['epoch']].append(entry['percent'])

    epochs = sorted(epoch_dict.keys())
    avg_pct = np.array([np.mean(epoch_dict[e], axis=0) for e in epochs])

    plt.figure(figsize=(6, 6))
    styles = ['-', '--', '-.', (0, (3, 1, 1, 1))]
    labels = ["MAE", "Percep", "SSIM", "Adv"]
    for i in range(4):
        plt.plot(epochs, avg_pct[:, i], linestyle=styles[i], color='black', label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Percent Contribution to Total Generator Loss (%)")
    plt.title("Relative Loss Share per Epoch")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# === Training Step ===
def train_step(
        model_G, model_D, optimizer_G, optimizer_D,
        cover_images, stego_images, rev_secret_img,
        diff_images, gt_images,
        perceptual_loss_fn,
        epoch, batch_idx, device,
        lambda_mae_dynamic, lambda_percep_dynamic,
        lambda_ssim_dynamic, lambda_adv_dynamic,
        out_dir
):
    model_G.train()
    model_D.train()
    print(f"[Epoch {epoch}] Batch {batch_idx}")

    # === Train Discriminator (only every 3rd batch) ===
    if batch_idx % 3 == 0:
        optimizer_D.zero_grad()
        with torch.no_grad():
            fake_imgs = model_G(diff_images.float(), stego_images.float()).detach()

        real_preds = model_D(gt_images.float())
        fake_preds = model_D(fake_imgs)

        d_loss_real = torch.mean(F.relu(1.0 - real_preds))
        d_loss_fake = torch.mean(F.relu(1.0 + fake_preds))
        loss_D = d_loss_real + d_loss_fake
        loss_D.backward()
        optimizer_D.step()
    else:
        loss_D = torch.tensor(0.0)

    # === Train Generator ===
    optimizer_G.zero_grad()
    outputs = model_G(diff_images.float(), stego_images.float())
    outputs = outputs.clamp(0.0, 1.0)
    gt_images = gt_images.clamp(0.0, 1.0)

    mae_per_image = torch.abs(outputs - gt_images).view(outputs.size(0), -1).mean(dim=1)
    G_mae_loss = mae_per_image.mean()
    # G_mae_loss = F.mse_loss(outputs, gt_images)
    G_perceptual_loss = perceptual_loss_fn(outputs, gt_images)
    G_ssim_loss = ssim_loss(outputs, gt_images)
    G_adv_loss = -model_D(outputs).mean()  # Hinge loss for G

    loss_G = (
            lambda_mae_dynamic * G_mae_loss +
            lambda_percep_dynamic * G_perceptual_loss +
            lambda_ssim_dynamic * G_ssim_loss +
            lambda_adv_dynamic * G_adv_loss
    )

    if not torch.isfinite(loss_G):
        print("NaN in generator loss. Skipping batch.")
        return model_G, model_D, None, None

    loss_G.backward()
    optimizer_G.step()

    # === Logging ===
    log_path = os.path.join(out_dir, 'loss_log_all.pkl')
    log_data = load_losslog_pkl(log_path)

    # Compute weighted components
    mae_contrib = lambda_mae_dynamic * G_mae_loss.item()
    percep_contrib = lambda_percep_dynamic * G_perceptual_loss.item()
    ssim_contrib = lambda_ssim_dynamic * G_ssim_loss.item()
    adv_contrib = lambda_adv_dynamic * G_adv_loss.item()
    total = mae_contrib + percep_contrib + ssim_contrib + adv_contrib + 1e-8  # avoid div-by-zero

    log_data.append({
        'epoch': epoch,
        'batch': batch_idx,
        'losses': [G_mae_loss.item(), G_perceptual_loss.item(), G_ssim_loss.item(), G_adv_loss.item()],
        'contrib': [mae_contrib, percep_contrib, ssim_contrib, adv_contrib],
        'percent': [
            100 * mae_contrib / total,
            100 * percep_contrib / total,
            100 * ssim_contrib / total,
            100 * adv_contrib / total
        ]
    })
    save_losslog_pkl(log_path, log_data)

    if batch_idx % 10 == 0:
        plot_loss_contributions(log_data, os.path.join(out_dir, 'loss_contrib_latest.png'))
        plot_loss_values(log_data, os.path.join(out_dir, 'loss_values_latest.png'))
        plot_loss_percentage(log_data, os.path.join(out_dir, 'loss_percent_latest.png'))

    return model_G, model_D, loss_G.item(), loss_D.item()


# === Generator Training Wrapper ===
def generator_network_train(model_G, model_D, optimizer_G, optimizer_D, scheduler_G, scheduler_D,
                            cover_image, container_image, rev_secret_img, diff_image, gt_image,
                            epoch, batch_idx, device, out_dir):
    RESULTS_TRAIN_PATH = os.path.join(out_dir, 'train')
    utils.check_dir([RESULTS_TRAIN_PATH])
    lambda_mae, lambda_percep, lambda_ssim, lambda_adv = get_lambda_schedule(epoch)
    perceptual_loss_fn = VGGPerceptualLoss(epoch=epoch, weight=1.0).to(device)
    print(batch_idx)

    return train_step(
        model_G, model_D, optimizer_G, optimizer_D,
        cover_image, container_image, rev_secret_img, diff_image, gt_image,
        perceptual_loss_fn,
        epoch, batch_idx, device,
        lambda_mae, lambda_percep, lambda_ssim, lambda_adv,
        RESULTS_TRAIN_PATH
    )
