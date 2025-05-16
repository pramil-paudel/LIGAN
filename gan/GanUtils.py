import PIL
import matplotlib
from PIL.Image import Image

matplotlib.use("Agg")
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from lensless.LenslessConversion import LenslessConversion
import gan.GanModels as G_models
from skimage.transform import resize
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.optim.lr_scheduler import LambdaLR

class NumpyDataset(Dataset):
    """docstring for NumpyDataset"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform


def check_dir(PATHS):
    for path in PATHS:
        if not os.path.exists(path):
            os.makedirs(path)
            print(path, 'created')
        # else:
        #     print(path, 'already exists')


def np_loader(PATH):
    sample = np.load(PATH, allow_pickle=False)
    return sample


def image_loader(PATH):
    sample = PIL.Image.open(PATH).convert('RGB')
    return sample


def normalize(I):
    I_norm = (I - np.amin(I)) / (np.amax(I) - np.amin(I))
    return I_norm


def preplot(image):
    # Handle (C, H, W) to (H, W, C) conversion if needed
    if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)

    # Resize to (64, 64)
    if image.ndim == 2:  # Grayscale (H, W)
        image_resized = resize(image, (64, 64), anti_aliasing=True, preserve_range=True)
        return image_resized
    elif image.ndim == 3 and image.shape[2] == 3:  # Color (H, W, 3)
        image_resized = resize(image, (64, 64), anti_aliasing=True, preserve_range=True)

        # Optional: swap channels if needed (BGR to RGB)
        image_color = np.zeros_like(image_resized)
        image_color[:, :, 0] = image_resized[:, :, 2]
        image_color[:, :, 1] = image_resized[:, :, 1]
        image_color[:, :, 2] = image_resized[:, :, 0]
        return image_color
    else:
        raise ValueError(f"Unexpected shape inside preplot(): {image.shape}")


def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(12, 8))
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator_MSE+Adv', alpha=0.5)
    plt.plot(losses.T[2], label='Generators_MSE', alpha=0.5)
    plt.plot(losses.T[3], label='Generator_Adv', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.close()  # plt.show()


def smooth_curve(data, smoothing=0.9):
    smoothed = []
    prev = data[0]
    for val in data:
        smoothed_val = smoothing * prev + (1 - smoothing) * val
        smoothed.append(smoothed_val)
        prev = smoothed_val
    return np.array(smoothed)


def plot_losses(losses, name='figure', RESULTS_DIR='./', method='GAN', smoothing=0.9):
    losses = np.asarray(losses)

    if losses.shape[1] == 9:
        # WGAN losses
        loss_D, loss_G, D_real, D_fake, G_mse, G_adv, wass_D, gp, min_val_loss = losses.T
        rows, cols = 3, 2
    elif losses.shape[1] == 7:
        # GAN losses
        loss_D, loss_G, D_real, D_fake, G_mse, G_adv, min_val_loss = losses.T
        rows, cols = 2, 2
    else:
        raise ValueError(f"Unexpected number of loss values per entry: {losses.shape[1]}")

    x = np.arange(len(loss_D))

    # Apply smoothing
    loss_D_smooth = smooth_curve(loss_D, smoothing)
    loss_G_smooth = smooth_curve(loss_G, smoothing)
    D_real_smooth = smooth_curve(D_real, smoothing)
    D_fake_smooth = smooth_curve(D_fake, smoothing)
    G_mse_smooth = smooth_curve(G_mse, smoothing)
    G_adv_smooth = smooth_curve(G_adv, smoothing)
    if losses.shape[1] == 9:
        wass_D_smooth = smooth_curve(wass_D, smoothing)
        gp_smooth = smooth_curve(gp, smoothing)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    axes[0].plot(x, loss_D, label='Raw', alpha=0.3)
    axes[0].plot(x, loss_D_smooth, label='Smoothed', linewidth=2)
    axes[0].set_title('Discriminator Loss')
    axes[0].set_ylabel('$L_{disc}$')
    axes[0].set_xlabel('Batch')
    axes[0].legend()

    axes[1].plot(x, D_real, label='D_real Raw', alpha=0.3)
    axes[1].plot(x, D_fake, label='D_fake Raw', alpha=0.3)
    axes[1].plot(x, D_real_smooth, label='D_real Smooth', linewidth=2)
    axes[1].plot(x, D_fake_smooth, label='D_fake Smooth', linewidth=2)
    axes[1].set_title('Discriminator Outputs')
    axes[1].set_ylabel('$D(I)$')
    axes[1].set_xlabel('Batch')
    axes[1].legend()

    axes[2].plot(x, loss_G, label='Raw', alpha=0.3)
    axes[2].plot(x, loss_G_smooth, label='Smoothed', linewidth=2)
    axes[2].set_title('Generator Loss')
    axes[2].set_ylabel('$L_{gen}$')
    axes[2].set_xlabel('Batch')
    axes[2].legend()

    axes[3].plot(x, G_mse, label='MSE Raw', alpha=0.3)
    axes[3].plot(x, G_adv, label='Adv Raw', alpha=0.3)
    axes[3].plot(x, G_mse_smooth, label='MSE Smooth', linewidth=2)
    axes[3].plot(x, G_adv_smooth, label='Adv Smooth', linewidth=2)
    axes[3].set_title('Generator Component Losses')
    axes[3].set_ylabel('Loss Value')
    axes[3].set_xlabel('Batch')
    axes[3].legend()

    if losses.shape[1] == 9:
        axes[4].plot(x, wass_D, label='Raw', alpha=0.3)
        axes[4].plot(x, wass_D_smooth, label='Smoothed', linewidth=2)
        axes[4].set_title('Wasserstein Distance')
        axes[4].set_ylabel('Distance')
        axes[4].set_xlabel('Batch')
        axes[4].legend()

        axes[5].plot(x, gp, label='Raw', alpha=0.3)
        axes[5].plot(x, gp_smooth, label='Smoothed', linewidth=2)
        axes[5].set_title('Gradient Penalty')
        axes[5].set_ylabel('Penalty')
        axes[5].set_xlabel('Batch')
        axes[5].legend()

    fig.suptitle(f"Training Losses ({method})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(RESULTS_DIR, name + '.png')
    plt.savefig(save_path)
    plt.close(fig)

    print(f"[plot_losses] Saved plot: {save_path}")


# ############################################
#     LOAD DATASET
# ############################################
def load_data(DATA_DIR, batch_size, num_workers, phase):
    # Directories for train and test data
    TRAIN_ROOT = DATA_DIR + 'train/'
    TEST_ROOT = DATA_DIR + 'test/'

    processing_image_size = processing_image_size = (128, 128)
    # convert data to torch.FloatTensor
    lensless_transform = transforms.Compose(
        [
            transforms.Resize(size=processing_image_size),
            # transforms.RandomCrop(224),
            LenslessConversion(),
            transforms.ToTensor(),
        ]
    )

    normal_transform = transforms.Compose(
        [
            transforms.Resize(size=processing_image_size),
            transforms.ToTensor(),
        ]
    )
    # load the training and test datasets
    if phase == 'train' or phase == 'traintest' or phase == 'debug_train':
        # Load Training dataset
        train_data = datasets.DatasetFolder(
            root=TRAIN_ROOT,
            loader=image_loader,
            transform=lensless_transform,
            extensions=('.png', '.jpg'))

        train_val_data = datasets.DatasetFolder(
            root=TRAIN_ROOT,
            loader=image_loader,
            transform=normal_transform,
            extensions=('.png', '.jpg'))

        test_data = datasets.DatasetFolder(
            root=TEST_ROOT,
            loader=image_loader,
            transform=lensless_transform,
            extensions=('.png', '.jpg'))

        test_val_data = datasets.DatasetFolder(
            root=TEST_ROOT,
            loader=image_loader,
            transform=normal_transform,
            extensions=('.png', '.jpg'))

        print('Train Classes:', train_data.classes)
        print('# of TRAIN INPUT images:', len(train_data))
        print('# of TRAIN GT images:', len(train_val_data))

        print('Test Classes:', train_data.classes)
        print('# of TEST INPUT images:', len(test_data))
        print('# of TEST GT images:', len(test_val_data))

        # Split Input and GT Training images
        # This only loads single batch to debug training
        train_data_s = torch.utils.data.Subset(train_data, list(range(len(train_data))))
        train_val_data_s = torch.utils.data.Subset(train_val_data, list(range(len(train_val_data))))
        test_data_s = torch.utils.data.Subset(test_data, list(range(len(test_data))))
        test_val_data_s = torch.utils.data.Subset(test_val_data, list(range(len(test_val_data))))
        # Loaders
        train_data_loader = DataLoader(train_data_s, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        train_val_data_loader = DataLoader(train_val_data_s, batch_size=batch_size, num_workers=num_workers,
                                           shuffle=False)
        test_data_loader = DataLoader(test_data_s, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        test_val_data_loader = DataLoader(test_val_data_s, batch_size=batch_size, num_workers=num_workers,
                                          shuffle=False)
        print('Train Classes:', train_data.classes)
        print('# of TRAIN INPUT images:', len(train_data))
        print('# of TRAIN GT images:', len(train_val_data))

        print('Test Classes:', train_data.classes)
        print('# of TEST INPUT images:', len(test_data))
        print('# of TEST GT images:', len(test_val_data))
        return train_data_loader, train_val_data_loader, test_data_loader, test_val_data_loader


# ############################################
#     INITIALIZE/LOAD MODELS
# ############################################
def load_models(netG, netD, chkpoint, device, test_mode, ngpu, test_directory):
    # ====================
    # Training Options
    # ====================
    last_epoch = 0  # Initialize lastepoch
    losses = []  # Initialize Epoch losses
    iter_losses = []  # Initialize iteration losses
    # Initialize Generator and Discriminator Models
    model_G = G_models.define_G(netG)
    model_D = G_models.define_D(netD)
    model_G = model_G.to(device)
    model_D = model_D.to(device)

    def lr_schedule_G(epoch):
        if epoch < 5:
            return (epoch + 1) / 5.0
        progress = min((epoch - 5) / 95, 1.0)
        return 0.5 * (1 + np.cos(np.pi * progress))

    def lr_schedule_D(epoch):
        if epoch < 10:
            return 1.5  # Boosted learning rate for D
        elif epoch < 30:
            return 1.0
        else:
            return 0.5  # Slow decay for later stability

    # === Optimizer and Scheduler Setup ===
    base_lr_G = 2e-4
    base_lr_D = 1e-5
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=base_lr_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=base_lr_D, betas=(0.5, 0.999))
    scheduler_G = LambdaLR(optimizer_G, lr_lambda=lr_schedule_G)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda=lr_schedule_D)

    # learning_rate_G = 1e-4
    # learning_rate_D = 1e-4  # OR even 5e-5 if you want D a bit weaker
    #
    # optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
    # optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))
    #
    # scheduler_G = None  # REMOVE StepLR
    # scheduler_D = None  # REMOVE StepLR

    if test_mode == "test":
        if chkpoint != -1: last_epoch = chkpoint
        last_checkpoint = torch.load(test_directory + 'GAN.pt')
        model_G.load_state_dict(last_checkpoint['netG'])
        model_D.load_state_dict(last_checkpoint['netD'])
        optimizer_G.load_state_dict(last_checkpoint['optimizerG'])
        optimizer_D.load_state_dict(last_checkpoint['optimizerD'])
        scheduler_G.load_state_dict(last_checkpoint['schedulerG'])
        scheduler_D.load_state_dict(last_checkpoint['schedulerD'])
        losses = last_checkpoint['losses']
        iter_losses = last_checkpoint['iter_losses']
        print(optimizer_G.state_dict);
        print()
        print('SCHEDULER STATE_DICT')
        for var_name in scheduler_G.state_dict():
            print(var_name, "\t", scheduler_G.state_dict()[var_name])
        print()

    # ====================
    # Send to GPU
    # ====================
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):  # COMMENT OUT IF
        model_G = nn.DataParallel(model_G, list(range(ngpu)))  # DATA WAS SAVED
        model_D = nn.DataParallel(model_D, list(range(ngpu)))  # AS DATA_PARALLEL

    return model_G, model_D, optimizer_G, optimizer_D, scheduler_G, scheduler_D, losses, iter_losses, last_epoch


def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, device, LAMBDA=10):
    assert real_data.shape[1] == 3 and fake_data.shape[1] == 3

    # Resize real_data to match fake_data spatially
    if real_data.shape[2:] != fake_data.shape[2:]:
        real_data = F.interpolate(real_data, size=fake_data.shape[2:], mode='bilinear', align_corners=False)

    alpha = torch.rand(BATCH_SIZE, 1, 1, 1, device=device)
    alpha = alpha.expand(BATCH_SIZE, 3, fake_data.shape[2], fake_data.shape[3])

    interpolates = (alpha * real_data + (1 - alpha) * fake_data).to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates.float())

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def tensor_to_image(tensor):
    img = tensor.detach().cpu().numpy()

    # Handle channel-first 3D tensor
    if img.ndim == 3:
        if img.shape[0] == 3:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # → (H, W, C)
        elif img.shape[0] == 1:  # Grayscale (1, H, W)
            img = np.squeeze(img, axis=0)  # → (H, W)
    elif img.ndim == 4:
        raise ValueError("Expected a single image tensor, got a batch.")

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    if img.ndim == 2:
        return Image.fromarray(img, mode='L')  # Grayscale
    elif img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(img, mode='RGB')  # Color
    else:
        raise ValueError(f"Unexpected image shape for conversion: {img.shape}")


def save_training_grid(images, outputs, target, epoch, save_dir, batch_size):
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(5, batch_size)
    images_np = images[:num_samples].detach().cpu().numpy()
    outputs_np = outputs[:num_samples].detach().cpu().numpy()
    target_np = target[:num_samples].detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=3, ncols=num_samples, figsize=(20, 10), dpi=50)
    row_titles = ['Input', 'Output (Reconstructed)', 'Target (GT)']
    image_sets = [images_np, outputs_np, target_np]

    for row_idx, (img_set, row) in enumerate(zip(image_sets, axes)):
        for col_idx, (img, ax) in enumerate(zip(img_set, row)):
            img = img.astype(np.float32)  # << important!
            if img.ndim == 3:
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[0] == 1:
                    img = img.squeeze(0)
            img = resize(img, (64, 64), anti_aliasing=True, preserve_range=True)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img)
            ax.axis('off')
        row[0].set_ylabel(row_titles[row_idx], rotation=90, size='large')

    grid_path = os.path.join(save_dir, f"{epoch}_train_visualization.png")
    plt.tight_layout()
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')

    del images_np, outputs_np, target_np, fig, axes
    torch.cuda.empty_cache()
    print(f"Training image grid saved at: {grid_path}")


def plot_loss_contributions(contrib_logs, name='loss_contributions', RESULTS_DIR='./', smoothing_window=5):
    """
    contrib_logs: List or numpy array with columns:
    [G_ID contrib, G_MAE contrib, G_Percep contrib, G_SSIM contrib, G_Adv contrib]
    """
    contrib_logs = np.asarray(contrib_logs)
    epochs = np.arange(len(contrib_logs))

    # Normalize to get percentage contributions
    total_loss = np.sum(contrib_logs, axis=1, keepdims=True)
    percentage = 100 * contrib_logs / total_loss

    # Smooth the curves using moving average
    def smooth(y, window):
        if len(y) < window:
            return y
        return np.convolve(y, np.ones(window) / window, mode='valid')

    labels = ['Identity', 'MAE', 'Perceptual', 'SSIM', 'Adversarial']
    colors = ['black', 'dimgray', 'gray', 'darkgray', 'lightgray']  # Grayscale shades

    plt.figure(figsize=(10, 6))

    for i in range(percentage.shape[1]):
        smooth_percentage = smooth(percentage[:, i], smoothing_window)
        # Adjust x-axis because convolution reduces length by (window - 1)
        smooth_epochs = epochs[:len(smooth_percentage)]
        plt.plot(smooth_epochs, smooth_percentage, label=labels[i],
                 color=colors[i], linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Contribution (%)')
    plt.title('Generator Loss Contribution Over Time (Smoothed)')
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, name + '.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[plot_loss_contributions] Saved smoothed plot: {save_path}")
