import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class CleanDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2, dropout_rate=0.4):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class RevealNetDualInput(nn.Module):
    def __init__(self, nc=3, nhf=128, output_function=nn.Sigmoid, use_center_weight=True):
        super().__init__()
        self.use_center_weight = use_center_weight

        # Expecting two inputs, concatenate on channel dimension -> 6 channels
        self.initial_layer = nn.Sequential(
            nn.Conv2d(nc * 2, nhf, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(nhf),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(nhf, nhf * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(nhf * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(nhf * 2, nhf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nhf * 4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(nhf * 4, nhf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nhf * 8),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(nhf * 8),
            ResidualBlock(nhf * 8)
        )

        self.after_bottleneck_dropout = nn.Dropout2d(0.2)

        self.decoder1 = CleanDecoderBlock(nhf * 8, nhf * 4)
        self.decoder2 = CleanDecoderBlock(nhf * 4, nhf * 2)
        self.decoder3 = CleanDecoderBlock(nhf * 2, nhf)

        self.output_layer = nn.Sequential(
            nn.Conv2d(nhf, nc, kernel_size=7, stride=1, padding=3),
            output_function()
        )

        self.correction_block = nn.Sequential(
            nn.Conv2d(nc * 2, nc * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc * 2, nc, kernel_size=3, padding=1)
        )

        self.central_weight_layer = nn.Conv2d(nc, nc, kernel_size=1)
        nn.init.constant_(self.central_weight_layer.weight, 3.0)

    def forward(self, container_img, guidance_input):
        # Step 1: Concatenate both inputs (assume both are 3-channel images)
        x = torch.cat([container_img, guidance_input], dim=1)

        # Step 2: Go through encoder-decoder path
        x1 = self.initial_layer(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x4 = self.residual_blocks(x4)
        x4 = self.after_bottleneck_dropout(x4)

        x5 = self.decoder1(x4) + x3
        x6 = self.decoder2(x5) + x2
        x7 = self.decoder3(x6) + x1

        output = self.output_layer(x7)

        # Step 3: Add correction using the original guidance (e.g., partial recon)
        correction_input = torch.cat([output, guidance_input], dim=1)
        residual = self.correction_block(correction_input)
        output = output + residual  # Final corrected output

        # Step 4: Optional center weighting
        if self.use_center_weight and self.training:
            weighted = self.central_weight_layer(output)
            output = output + weighted.clamp(min=-1.0, max=3.0)

        return output
