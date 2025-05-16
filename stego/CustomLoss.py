import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16


def gaussian_window(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, channel=3, size_average=True):
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map


# 2. Define Combined Loss Function (MSE + L1 + SSIM)
class CombinedLoss(nn.Module):
    def __init__(self, factors=(0.5, 0.3, 0.2)):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.factors = factors  # Weights for MSE, L1, SSIM

    def forward(self, input_img, target_img):
        # Calculate individual losses
        mse = self.mse_loss(input_img, target_img)
        l1 = self.l1_loss(input_img, target_img)
        ssim_loss = 1 - ssim(input_img, target_img)  # SSIM loss (1 - SSIM)
        # Weighted sum of losses
        total_loss = (self.factors[0] * mse +
                      self.factors[1] * l1 +
                      self.factors[2] * ssim_loss)

        return total_loss


class CenterPriorityMAELoss(nn.Module):
    def __init__(self, center_size=(64, 64), weight=3):
        super(CenterPriorityMAELoss, self).__init__()
        self.center_size = center_size
        self.weight = weight

    def forward(self, y_true, y_pred):
        # Calculate absolute difference
        abs_diff = torch.abs(y_true - y_pred)

        # Get image dimensions (expecting 128x128)
        height, width = y_true.shape[2], y_true.shape[3]

        # Define the center region
        center_h_start = (height - self.center_size[0]) // 2
        center_w_start = (width - self.center_size[1]) // 2
        center_h_end = center_h_start + self.center_size[0]
        center_w_end = center_w_start + self.center_size[1]

        # Create a weighting mask
        mask = torch.ones_like(y_true)
        mask[:, :, center_h_start:center_h_end, center_w_start:center_w_end] *= self.weight

        # Apply the weighting mask
        weighted_abs_diff = abs_diff * mask

        # Calculate the mean absolute error
        return torch.mean(weighted_abs_diff)


class CenterPriorityMAELossTwoBoxes(nn.Module):
    def __init__(self, center_size=(64, 64), outer_box_size=(80, 80), weight_inner=4, weight_outer=1.0):
        super(CenterPriorityMAELossTwoBoxes, self).__init__()
        self.center_size = center_size
        self.outer_box_size = outer_box_size
        self.weight_inner = weight_inner
        self.weight_outer = weight_outer

    def forward(self, y_true, y_pred):
        # Calculate absolute difference
        abs_diff = torch.abs(y_true - y_pred)

        # Get image dimensions (expecting 128x128)
        height, width = y_true.shape[2], y_true.shape[3]

        # Define the inner center region (64x64)
        center_h_start = (height - self.center_size[0]) // 2
        center_w_start = (width - self.center_size[1]) // 2
        center_h_end = center_h_start + self.center_size[0]
        center_w_end = center_w_start + self.center_size[1]

        # Define the outer box region (80x80)
        outer_h_start = (height - self.outer_box_size[0]) // 2
        outer_w_start = (width - self.outer_box_size[1]) // 2
        outer_h_end = outer_h_start + self.outer_box_size[0]
        outer_w_end = outer_w_start + self.outer_box_size[1]

        # Create a weighting mask initialized to zero (ignore pixels outside 80x80)
        mask = torch.zeros_like(y_true)

        # Apply weights to the inner 64x64 region
        mask[:, :, center_h_start:center_h_end, center_w_start:center_w_end] = self.weight_inner

        # Apply weights to the outer 80x80 region (excluding inner 64x64 region)
        mask[:, :, outer_h_start:outer_h_end, outer_w_start:outer_w_end] = self.weight_outer
        mask[:, :, center_h_start:center_h_end, center_w_start:center_w_end] = self.weight_inner

        # Apply the weighting mask
        weighted_abs_diff = abs_diff * mask

        # Calculate the mean absolute error, ignoring areas with zero weight
        return torch.sum(weighted_abs_diff) / torch.sum(mask)


class CenterPriorityMAEWithMaskedTVLoss(nn.Module):
    def __init__(self, center_size=(64, 64), outer_box_size=(100, 100),
                 weight_inner=2, weight_outer=1.0, weight_background=0.05,
                 tv_weight=1e-5, combined_weight=0.2):
        super(CenterPriorityMAEWithMaskedTVLoss, self).__init__()
        self.center_size = center_size
        self.outer_box_size = outer_box_size
        self.weight_inner = weight_inner
        self.weight_outer = weight_outer
        self.weight_background = weight_background  # very low weight outside 100x100
        self.tv_weight = tv_weight
        self.combined_weight = combined_weight

        # Your combined loss
        self.combined_loss_fn = CombinedLoss(factors=(0.5, 0.5, 0.0))  # placeholder

    def _generate_mask(self, shape, device):
        _, _, H, W = shape
        mask = torch.full((1, 1, H, W), self.weight_background, device=device)

        # Define inner 64x64 region
        center_h_start = (H - self.center_size[0]) // 2
        center_w_start = (W - self.center_size[1]) // 2
        center_h_end = center_h_start + self.center_size[0]
        center_w_end = center_w_start + self.center_size[1]

        # Define outer 100x100 region
        outer_h_start = (H - self.outer_box_size[0]) // 2
        outer_w_start = (W - self.outer_box_size[1]) // 2
        outer_h_end = outer_h_start + self.outer_box_size[0]
        outer_w_end = outer_w_start + self.outer_box_size[1]

        # Assign weights
        mask[:, :, outer_h_start:outer_h_end, outer_w_start:outer_w_end] = self.weight_outer
        mask[:, :, center_h_start:center_h_end, center_w_start:center_w_end] = self.weight_inner

        return mask

    def forward(self, y_true, y_pred):
        mask = self._generate_mask(y_true.shape, y_true.device)
        abs_diff = torch.abs(y_true - y_pred)

        # Weighted MAE
        weighted_abs_diff = abs_diff * mask
        center_priority_mae = torch.sum(weighted_abs_diff) / torch.sum(mask)

        # Smooth TV loss everywhere but weighted by mask
        diff_h = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])
        diff_v = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])
        tv_loss = torch.sum(diff_h * mask[:, :, :, :-1]) + torch.sum(diff_v * mask[:, :, :-1, :])

        # Combined loss on 64x64 center only
        center_h_start = (y_true.shape[2] - self.center_size[0]) // 2
        center_w_start = (y_true.shape[3] - self.center_size[1]) // 2
        center_h_end = center_h_start + self.center_size[0]
        center_w_end = center_w_start + self.center_size[1]
        masked_y_true = y_true[:, :, center_h_start:center_h_end, center_w_start:center_w_end]
        masked_y_pred = y_pred[:, :, center_h_start:center_h_end, center_w_start:center_w_end]
        combined_loss = self.combined_loss_fn(masked_y_pred, masked_y_true)

        total_loss = (
                center_priority_mae
                + self.tv_weight * tv_loss
                + self.combined_weight * combined_loss
        )
        return total_loss


class BalancedCenterLoss(nn.Module):
    def __init__(self, gaussian_sigma=0.5, weight_inner=4, weight_background=0.8, tv_weight=1e-5):
        super(BalancedCenterLoss, self).__init__()
        self.gaussian_sigma = gaussian_sigma
        self.weight_inner = weight_inner
        self.weight_background = weight_background
        self.tv_weight = tv_weight

    def _generate_gaussian_mask(self, H, W, device):
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device))
        d = torch.sqrt(x ** 2 + y ** 2)
        gaussian = torch.exp(-((d / self.gaussian_sigma) ** 2))
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]
        gaussian = gaussian / gaussian.max()

        # Blend between background and center weights
        mask = self.weight_background + (self.weight_inner - self.weight_background) * gaussian
        return mask

    def forward(self, y_true, y_pred):
        # Resize y_true if needed to match prediction size
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true, size=y_pred.shape[2:], mode='bilinear', align_corners=True)

        _, _, H, W = y_true.shape
        mask = self._generate_gaussian_mask(H, W, y_true.device)

        abs_diff = torch.abs(y_true - y_pred)
        weighted_abs_diff = abs_diff * mask
        mae_loss = torch.sum(weighted_abs_diff) / (torch.sum(mask) + 1e-8)

        diff_h = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])
        diff_v = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])
        tv_loss = torch.sum(diff_h) + torch.sum(diff_v)

        # total_loss = mae_loss
        total_loss = mae_loss + self.tv_weight * tv_loss
        return total_loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(3, 8, 15), weight=0.2):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg.children())[:max(layers) + 1])
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.selected_layers = layers
        self.weight = weight

    def forward(self, x, y):
        x = (x - 0.5) * 2  # Normalize to [-1, 1]
        y = (y - 0.5) * 2
        loss = 0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.selected_layers:
                loss += F.l1_loss(x, y)
        return self.weight * loss


class BalancedCenterLossSecond(nn.Module):
    def __init__(self, gaussian_sigma=0.3, weight_inner=5, weight_background=0.8, tv_weight=1e-5,
                 perceptual_weight=0.2):
        super(BalancedCenterLossSecond, self).__init__()
        self.gaussian_sigma = gaussian_sigma
        self.weight_inner = weight_inner
        self.weight_background = weight_background
        self.tv_weight = tv_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss_fn = VGGPerceptualLoss(weight=perceptual_weight) if perceptual_weight > 0 else None

    def _generate_gaussian_mask(self, H, W, device):
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device), indexing='ij')
        d = torch.sqrt(x ** 2 + y ** 2)
        gaussian = torch.exp(-((d / self.gaussian_sigma) ** 2))
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)
        gaussian = gaussian / gaussian.max()
        mask = self.weight_background + (self.weight_inner - self.weight_background) * gaussian
        return mask

    def forward(self, y_pred, y_true, perceptual_target=None):
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true, size=y_pred.shape[2:], mode='bilinear', align_corners=True)

        _, _, H, W = y_true.shape
        mask = self._generate_gaussian_mask(H, W, y_true.device)

        abs_diff = torch.abs(y_true - y_pred)
        weighted_abs_diff = abs_diff * mask
        mae_loss = torch.sum(weighted_abs_diff) / (torch.sum(mask) + 1e-8)

        diff_h = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])
        diff_v = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])
        tv_loss = torch.sum(diff_h) + torch.sum(diff_v)

        total_loss = mae_loss + self.tv_weight * tv_loss

        if self.perceptual_loss_fn is not None and perceptual_target is not None:
            total_loss += self.perceptual_loss_fn(y_pred, perceptual_target)

        return total_loss


class VGGPerceptualLossSecond(nn.Module):
    def __init__(self, layers=(1, 3, 8, 15), layer_weights=None, weight=0.2):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg.children())[:max(layers) + 1])
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.selected_layers = layers
        self.layer_weights = layer_weights or {1: 1.0, 3: 0.8, 8: 0.5, 15: 0.3}
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
            if i in self.selected_layers:
                loss += self.layer_weights.get(i, 1.0) * F.l1_loss(x, y)
        return self.weight * loss


class CenterFocusedLoss(nn.Module):
    def __init__(self,
                 square_size=100,
                 weight_inner=5.0,
                 weight_background=0.5,
                 tv_weight=1e-5,
                 perceptual_weight=0.2,
                 global_l1_weight=0.05):
        super().__init__()
        self.square_size = square_size
        self.weight_inner = weight_inner
        self.weight_background = weight_background
        self.tv_weight = tv_weight
        self.global_l1_weight = global_l1_weight
        self.perceptual_weight = perceptual_weight

        self.last_weighted_l1 = 0
        self.last_tv = 0
        self.last_perceptual = 0
        self.last_global_l1 = 0

    def _make_soft_center_mask(self, H, W, device):
        y = torch.linspace(-1, 1, H, device=device).view(-1, 1)
        x = torch.linspace(-1, 1, W, device=device).view(1, -1)
        dist = torch.sqrt(x ** 2 + y ** 2)
        gaussian = torch.exp(-((dist * (H / self.square_size)) ** 2))
        mask = self.weight_background + (self.weight_inner - self.weight_background) * gaussian
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, y_pred, y_true):
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true, size=y_pred.shape[2:], mode='bilinear', align_corners=True)

        _, _, H, W = y_true.shape
        mask = self._make_soft_center_mask(H, W, y_true.device)

        abs_diff = torch.abs(y_true - y_pred)
        weighted_loss = torch.sum(abs_diff * mask) / (torch.sum(mask) + 1e-8)

        global_l1 = F.l1_loss(y_pred, y_true)

        tv_h = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])
        tv_v = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])
        tv_loss = torch.sum(tv_h) + torch.sum(tv_v)

        total_loss = (
                weighted_loss +
                self.tv_weight * tv_loss +
                self.global_l1_weight * global_l1
        )

        self.last_weighted_l1 = weighted_loss.item()
        self.last_global_l1 = global_l1.item()
        self.last_tv = tv_loss.item()

        return total_loss


class SimplifiedLoss(nn.Module):
    def __init__(self, perceptual_weight=0.4, tv_weight=1e-5):
        super().__init__()
        self.perceptual_loss = VGGPerceptualLossSecond(weight=perceptual_weight)
        self.tv_weight = tv_weight

    def forward(self, y_pred, y_true):
        l1 = F.l1_loss(y_pred, y_true)

        # Total Variation (smoothing)
        tv = torch.mean(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) + \
             torch.mean(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]))

        # Perceptual (VGG-based)
        perceptual = self.perceptual_loss(y_pred, y_true)

        # Combine
        return l1 + self.tv_weight * tv + perceptual


class VGGPerceptualLossThree(nn.Module):
    def __init__(self, weight=1.0, center_crop_size=64):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.weight = weight
        self.center_crop_size = center_crop_size

    def _crop_center(self, x):
        _, _, H, W = x.shape
        crop = self.center_crop_size
        start_h = (H - crop) // 2
        start_w = (W - crop) // 2
        return x[:, :, start_h:start_h + crop, start_w:start_w + crop]

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        # Normalize to [-1, 1] (VGG expects this range)
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2

        # Crop to center 64x64
        x_cropped = self._crop_center(x)
        y_cropped = self._crop_center(y)

        # VGG perceptual loss
        perceptual_loss = F.l1_loss(self.vgg_layers(x_cropped), self.vgg_layers(y_cropped))
        return self.weight * perceptual_loss


class CenterFocusedLossTwo(nn.Module):
    def __init__(self,
                 square_size=64,
                 weight_inner=12.0,
                 weight_background=0.05,
                 perceptual_weight=0.4,
                 global_l1_weight=0.1,
                 center_mae_weight=1.0,
                 debug=False):
        super().__init__()
        self.square_size = square_size
        self.weight_inner = weight_inner
        self.weight_background = weight_background
        self.perceptual_weight = perceptual_weight
        self.global_l1_weight = global_l1_weight
        self.center_mae_weight = center_mae_weight
        self.debug = debug

        self.vgg_loss = VGGPerceptualLossThree(weight=1.0) if perceptual_weight > 0 else None

    def _make_soft_center_mask(self, H, W, device):
        y = torch.linspace(-1, 1, H, device=device).view(-1, 1)
        x = torch.linspace(-1, 1, W, device=device).view(1, -1)
        dist = torch.sqrt(x ** 2 + y ** 2)
        gaussian = torch.exp(-((dist * (H / self.square_size)) ** 2))
        mask = self.weight_background + (self.weight_inner - self.weight_background) * gaussian
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, y_pred, y_true):
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true, size=y_pred.shape[2:], mode='bilinear', align_corners=True)

        _, _, H, W = y_true.shape
        mask = self._make_soft_center_mask(H, W, y_true.device)

        # Center-focused MAE
        abs_diff = torch.abs(y_true - y_pred)
        weighted_loss = torch.sum(abs_diff * mask) / (torch.sum(mask) + 1e-8)
        center_mae = self.center_mae_weight * weighted_loss

        # Global L1 loss
        global_l1 = self.global_l1_weight * F.l1_loss(y_pred, y_true)

        # Perceptual Loss
        perceptual = self.perceptual_weight * self.vgg_loss(y_pred, y_true) if self.vgg_loss else 0.0

        total = center_mae + global_l1 + perceptual

        if self.debug:
            print(
                f"[CenterFocusedLossTwo] center_mae={center_mae.item():.4f}, global_l1={global_l1.item():.4f}, perceptual={perceptual.item() if isinstance(perceptual, torch.Tensor) else 0:.4f}")

        return total


class VGGPerceptualLossFour(nn.Module):
    def __init__(self, weight=1.0, center_crop_size=64):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.weight = weight
        self.center_crop_size = center_crop_size

    def _crop_center(self, x):
        _, _, H, W = x.shape
        crop = self.center_crop_size
        start_h = (H - crop) // 2
        start_w = (W - crop) // 2
        return x[:, :, start_h:start_h + crop, start_w:start_w + crop]

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2
        x_cropped = self._crop_center(x)
        y_cropped = self._crop_center(y)
        perceptual_loss = F.l1_loss(self.vgg_layers(x_cropped), self.vgg_layers(y_cropped))
        return self.weight * perceptual_loss


class CenterFocusedLossThree(nn.Module):
    def __init__(self,
                 square_size=64,
                 weight_inner=12.0,
                 weight_background=0.05,
                 perceptual_weight=0.4,
                 global_l1_weight=0.1,
                 center_mae_weight=1.0,
                 debug=False):
        super().__init__()
        self.square_size = square_size
        self.weight_inner = weight_inner
        self.weight_background = weight_background
        self.perceptual_weight = perceptual_weight
        self.global_l1_weight = global_l1_weight
        self.center_mae_weight = center_mae_weight
        self.debug = debug

        self.vgg_loss = VGGPerceptualLossFour(weight=1.0) if perceptual_weight > 0 else None

    def _make_soft_center_mask(self, H, W, device):
        y = torch.linspace(-1, 1, H, device=device).view(-1, 1)
        x = torch.linspace(-1, 1, W, device=device).view(1, -1)
        dist = torch.sqrt(x ** 2 + y ** 2)
        gaussian = torch.exp(-((dist * (H / self.square_size)) ** 2))
        mask = self.weight_background + (self.weight_inner - self.weight_background) * gaussian
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, y_pred, y_true, perceptual_pred=None, perceptual_target=None):
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true, size=y_pred.shape[2:], mode='bilinear', align_corners=True)

        _, C, H, W = y_true.shape
        mask = self._make_soft_center_mask(H, W, y_true.device)

        # Center-weighted L1
        abs_diff = torch.abs(y_true - y_pred)
        weighted_loss = torch.sum(abs_diff * mask) / (torch.sum(mask) + 1e-8)
        center_mae = self.center_mae_weight * weighted_loss

        # Channel-wise Global L1
        channelwise_l1 = sum(F.l1_loss(y_pred[:, c], y_true[:, c]) for c in range(C))
        global_l1 = self.global_l1_weight * channelwise_l1

        # Perceptual Loss (optional)
        if self.vgg_loss is not None and perceptual_pred is not None and perceptual_target is not None:
            perceptual = self.perceptual_weight * self.vgg_loss(perceptual_pred, perceptual_target)
        else:
            perceptual = 0.0

        # ✅ Color Mean Loss
        mean_y_pred = y_pred.mean(dim=[2, 3])  # shape (B, C)
        mean_y_true = y_true.mean(dim=[2, 3])  # shape (B, C)
        color_mean_loss = 0.2 * F.mse_loss(mean_y_pred, mean_y_true)  # <-- tunable weight

        # Final Total Loss
        total = center_mae + global_l1 + perceptual + color_mean_loss

        if self.debug:
            print(
                f"[CenterFocusedLossThree] center_mae={center_mae.item():.4f}, global_l1={global_l1.item():.4f}, perceptual={perceptual.item() if isinstance(perceptual, torch.Tensor) else 0:.4f}, color_mean={color_mean_loss.item():.4f}")
        return total
