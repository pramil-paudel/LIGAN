import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


def compute_metrics(batch1, batch2, crop_border=0):
    """
    Compute PSNR, SSIM, and MSE between two batches of images.
    Assumes input tensors are in (B, C, H, W) and in range [0, 1].
    """
    batch1 = batch1.detach().cpu().numpy()
    batch2 = batch2.detach().cpu().numpy()

    results = []
    for img1, img2 in zip(batch1, batch2):
        # Transpose from (C, H, W) to (H, W, C)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

        # Optional border crop
        if crop_border > 0:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

        # Clamp just in case
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        # Compute metrics
        mse = np.mean((img1 - img2) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
        win_size = min(7, img1.shape[0], img1.shape[1])
        ssim_val = structural_similarity(img1, img2, data_range=1.0, channel_axis=-1, win_size=win_size)

        results.append({
            "PSNR": psnr,
            "SSIM": ssim_val,
            "MSE": mse
        })

    return results


def plot_grid_triplet_dual_metrics(cover_out, container_out, secret_out, reco_out, original_gt,
                                   num_samples=4, save_path=None, metrics=None):
    titles = ["Cover", "Container", "Secret", "Reconstructed", "GT", "Difference (Cont - Cov)"]
    tensors = [cover_out, container_out, secret_out, reco_out, original_gt]
    diff_tensors = container_out - cover_out
    tensors.append(diff_tensors)

    fig, axes = plt.subplots(nrows=7, ncols=num_samples, figsize=(4 * num_samples, 17), dpi=100)
    plt.subplots_adjust(hspace=0.3, wspace=0.05)

    def safe_tensor_to_image(t):
        t_np = t.detach().cpu().numpy()
        if t_np.ndim == 3 and t_np.shape[0] in [1, 3]:
            t_np = np.transpose(t_np, (1, 2, 0))
        elif t_np.ndim == 3:
            t_np = t_np
        else:
            t_np = np.squeeze(t_np)
        t_np = np.clip((t_np - t_np.min()) / (t_np.max() - t_np.min() + 1e-8), 0, 1)
        return t_np

    # Plot images
    for row_idx, (img_set, row_axes) in enumerate(zip(tensors, axes[:-1])):
        for col_idx, ax in enumerate(row_axes):
            ax.axis("off")
            if col_idx < len(img_set):
                img_np = safe_tensor_to_image(img_set[col_idx])
                ax.imshow(img_np)
            if row_idx == 0:
                ax.set_title(f"Sample {col_idx + 1}", fontsize=12)

    # Plot metrics text
    if metrics:
        for col_idx, ax in enumerate(axes[-1]):
            ax.axis("off")
            if col_idx < len(metrics):
                hide_text, reveal_text = metrics[col_idx]
                ax.text(0.5, 0.6, hide_text, fontsize=8, ha='center', va='top', wrap=True)
                ax.text(0.5, 0.1, reveal_text, fontsize=8, ha='center', va='top', wrap=True)

    # Row labels
    for i, label in enumerate(titles):
        axes[i, 0].annotate(label, xy=(0, 0.5), xytext=(-axes[i, 0].yaxis.labelpad - 30, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            ha='right', va='center', fontsize=11, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Dual-metric grid saved at: {save_path}")

    plt.close(fig)
    plt.close('all')  # <<< CRITICAL TO AVOID FIGURE BUILDUP

    # Final cleanup
    del fig, axes, tensors, diff_tensors
    import gc
    gc.collect()


def plot_losses(losses, save_dir=None, title_prefix="Training", show=True):
    if not losses:
        print("No losses to plot.")
        return

    first_loss = losses[0]
    epochs = np.arange(len(losses))

    if len(first_loss) == 9:
        # WGAN mode
        loss_D, loss_G, D_real, D_fake, G_mse, G_adv, wass_D, gp, min_val_loss = zip(*losses)
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f"{title_prefix} Loss Curves (WGAN)", fontsize=16)

        axes[0, 0].plot(epochs, loss_D, label="Discriminator Loss")
        axes[0, 1].plot(epochs, loss_G, label="Generator Loss", color='orange')
        axes[1, 0].plot(epochs, G_mse, label="Generator MSE Loss", color='green')
        axes[1, 1].plot(epochs, G_adv, label="Generator Adversarial Loss", color='red')
        axes[2, 0].plot(epochs, wass_D, label="Wasserstein Distance", color='purple')
        axes[2, 1].plot(epochs, gp, label="Gradient Penalty", color='brown')

        titles = ["Discriminator Loss", "Generator Loss", "Generator MSE Loss",
                  "Generator Adversarial Loss", "Wasserstein Distance", "Gradient Penalty"]

        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.grid()
            ax.set_xlabel("Epochs")
            ax.legend()

    elif len(first_loss) == 7:
        # GAN mode
        loss_D, loss_G, D_real, D_fake, G_mse, G_adv, min_val_loss = zip(*losses)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"{title_prefix} Loss Curves (GAN)", fontsize=16)

        axes[0, 0].plot(epochs, loss_D, label="Discriminator Loss")
        axes[0, 1].plot(epochs, loss_G, label="Generator Loss", color='orange')
        axes[1, 0].plot(epochs, G_mse, label="Generator MSE Loss", color='green')
        axes[1, 1].plot(epochs, G_adv, label="Generator Adversarial Loss", color='red')

        titles = ["Discriminator Loss", "Generator Loss", "Generator MSE Loss",
                  "Generator Adversarial Loss"]

        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.grid()
            ax.set_xlabel("Epochs")
            ax.legend()

    else:
        print("Unexpected losses format!")
        return

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "training_loss_plot.png")
        plt.savefig(plot_path)
        print(f"[plot_losses] Saved plot: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    del fig, axes
    import gc
    gc.collect()
