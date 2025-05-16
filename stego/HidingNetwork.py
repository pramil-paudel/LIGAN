import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------- Patch Embedding and Unembedding ---------------- #

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(
                f"Input too small: got ({H}x{W}) for patch size {self.patch_size}"
            )
        x = self.proj(x)  # (B, embed_dim, H//patch, W//patch)
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        return x_flat, H, W


class PatchUnembedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        if H is None or W is None:
            side = int(N ** 0.5)
            assert side * side == N, f"Cannot reshape {N} tokens to square image"
            H = W = side
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)


# ---------------- Swin Transformer Layer Placeholder ---------------- #

class SwinTransformerLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return x + self.attn(x)


# ---------------- Residual Swin Transformer Block (RSTB) ---------------- #

class RSTB(nn.Module):
    def __init__(self, embed_dim, num_layers=6, patch_size=4):
        super().__init__()
        self.blocks = nn.Sequential(*[SwinTransformerLayer(embed_dim) for _ in range(num_layers)])
        self.patch_unembed = PatchUnembedding()
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.patch_embed = PatchEmbedding(embed_dim, embed_dim, patch_size=patch_size)

    def forward(self, x, H, W):
        residual = x
        x = self.blocks(x)
        x_image = self.patch_unembed(x, H, W)
        x_image = self.conv(x_image)
        x, H_new, W_new = self.patch_embed(x_image)

        if residual.shape[1] != x.shape[1]:
            residual = residual[:, :x.shape[1], :]

        return x + residual, H_new, W_new


# ---------------- Full Hiding Network ---------------- #

class SwinHidingNet(nn.Module):
    def __init__(self, in_channels=6, mid_channels=96, out_channels=3, img_size=128, patch_size=4, num_blocks=4):
        super().__init__()
        self.patch_size = patch_size
        self.shallow = nn.Conv2d(in_channels, mid_channels, 3, padding=1)

        self.patch_embed = PatchEmbedding(mid_channels, mid_channels, patch_size=patch_size)
        self.transformer = nn.ModuleList([
            RSTB(mid_channels, patch_size=patch_size) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(mid_channels)
        self.patch_unembed = PatchUnembedding()

        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1)
        self.output = nn.Conv2d(mid_channels, out_channels, 3, padding=1)

    def forward(self, x):
        shallow_feat = self.shallow(x)  # (B, mid_channels, 128, 128)
        tokens, H, W = self.patch_embed(shallow_feat)

        for block in self.transformer:
            tokens, H, W = block(tokens, H, W)

        tokens = self.norm(tokens)
        deep_feat = self.patch_unembed(tokens, H, W)
        if deep_feat.shape[2:] != shallow_feat.shape[2:]:
            deep_feat = F.interpolate(deep_feat, size=shallow_feat.shape[2:], mode='bilinear', align_corners=False)

        fused = torch.cat([shallow_feat, deep_feat], dim=1)
        fused = self.fusion(fused)
        return self.output(fused)