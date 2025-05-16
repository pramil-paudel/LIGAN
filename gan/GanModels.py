import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def define_D(netD='basic'):
    if netD == 'basic':
        discriminator = Discriminator()
    elif netD == 'cycle':
        discriminator = Discriminator_Cycle()
    elif netD == 'wgan':
        discriminator = Discriminator_Cycle_WGAN()
    elif netD == 'patch':
        discriminator = PatchDiscriminatorHinge()
    return discriminator


def define_G(netG='unet'):
    if netG == "unet":
        generator = UNet((3, 64, 64))  # Same input shape
    elif netG == 'unetModified':
        generator = UNetModified((3, 64, 64))
    elif netG == 'GuidedRDN':
        generator = GuidedRDN128()  # Same input shape
    elif netG == 'RDN':
        generator = RDN(in_shape=(3, 64, 64))
    return generator


BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)  # Updated from F.upsample()
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x


class UNetModified(nn.Module):
    def __init__(self, in_shape):
        super(UNetModified, self).__init__()
        channels, height, width = in_shape

        # ---------------- Encoders ----------------
        self.down1 = StackEncoder(channels, 24, kernel_size=3)  # 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16

        # ---------------- Center (Residual Bottleneck) ----------------
        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1),
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1)
        )

        # ---------------- Decoders ----------------
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512

        # ---------------- Shallow Skips (for color preservation) ----------------
        # Project input channels to match decoder levels
        self.shallow_skip2 = nn.Conv2d(channels, 64, kernel_size=1)
        self.shallow_skip3 = nn.Conv2d(channels, 128, kernel_size=1)
        self.shallow_skip4 = nn.Conv2d(channels, 256, kernel_size=1)
        self.shallow_skip5 = nn.Conv2d(channels, 512, kernel_size=1)

        # ---------------- Classifier ----------------
        self.classify = nn.Conv2d(24, channels, kernel_size=1, bias=True)

    def forward(self, x):
        original_input = x  # Save for shallow skips

        # Encoder path
        down1, out = self.down1(x)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        # Center bottleneck
        out = self.center(out)

        # ----- Resize shallow skips to match each decoder level -----
        shallow5 = F.interpolate(self.shallow_skip5(original_input), size=down5.shape[2:], mode='bilinear',
                                 align_corners=False)
        shallow4 = F.interpolate(self.shallow_skip4(original_input), size=down4.shape[2:], mode='bilinear',
                                 align_corners=False)
        shallow3 = F.interpolate(self.shallow_skip3(original_input), size=down3.shape[2:], mode='bilinear',
                                 align_corners=False)
        shallow2 = F.interpolate(self.shallow_skip2(original_input), size=down2.shape[2:], mode='bilinear',
                                 align_corners=False)

        # Decoder path + normal skip + shallow skip
        out = self.up5(out, down5 + shallow5)
        out = self.up4(out, down4 + shallow4)
        out = self.up3(out, down3 + shallow3)
        out = self.up2(out, down2 + shallow2)
        out = self.up1(out, down1)  # down1 already low level, no shallow skip here

        out = self.classify(out)

        return out  # Do not squeeze. Keep channels intact


class UNet(nn.Module):
    def __init__(self, in_shape):
        super(UNet, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(channels, 24, kernel_size=3);  # 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, channels, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        # self.center = nn.Sequential(ConvBnRelu2d(256, 256, kernel_size=3, padding=1))

    def forward(self, x):
        out = x;
        down1, out = self.down1(out);
        down2, out = self.down2(out);
        down3, out = self.down3(out);
        down4, out = self.down4(out);
        down5, out = self.down5(out);

        out = self.center(out)
        out = self.up5(out, down5);
        out = self.up4(out, down4);
        out = self.up3(out, down3);
        out = self.up2(out, down2);
        out = self.up1(out, down1);

        out = self.classify(out);
        # out = torch.squeeze(out, dim=1);
        return out


class ResidualDenseBlockSmall(nn.Module):
    def __init__(self, in_channels, growth_channels=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(channels, growth_channels, kernel_size=3, padding=1))
            channels += growth_channels
        self.fusion = nn.Conv2d(channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for conv in self.layers:
            out = F.relu(conv(torch.cat(features, dim=1)))
            features.append(out)
        return x + self.fusion(torch.cat(features, dim=1))


class TinyRDNGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_features=32, num_blocks=3):
        super().__init__()
        self.shallow = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.rdb_blocks = nn.Sequential(*[
            ResidualDenseBlockSmall(num_features) for _ in range(num_blocks)
        ])
        self.fusion = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.output = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, secret_half, container_highres):
        # Downsample container to match secret size
        container_ds = F.interpolate(container_highres, size=secret_half.shape[2:], mode='bilinear',
                                     align_corners=False)
        x = torch.cat([secret_half, container_ds], dim=1)
        feat = self.shallow(x)
        rdb = self.rdb_blocks(feat)
        fused = self.fusion(rdb + feat)
        return self.output(fused)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.growth_channels = growth_channels
        channels = in_channels

        for _ in range(num_layers):
            self.layers.append(
                nn.Conv2d(channels, growth_channels, kernel_size=3, padding=1)
            )
            channels += growth_channels

        self.local_fusion = nn.Conv2d(channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for conv in self.layers:
            out = F.relu(conv(torch.cat(features, dim=1)))
            features.append(out)
        fused = self.local_fusion(torch.cat(features, dim=1))
        return x + fused


class GuidedRDN(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_features=64, num_blocks=6,
                 growth_channels=32, num_layers=5, refine=True):
        super().__init__()
        self.refine = refine
        self.shallow_feat = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.rdb_blocks = nn.Sequential(*[
            ResidualDenseBlock(num_features, growth_channels, num_layers)
            for _ in range(num_blocks)
        ])
        self.global_fusion = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

        if refine:
            self.refiner = ResidualRefiner(in_channels=out_channels)

    def forward(self, secret_half, container_highres):
        container_ds = F.interpolate(container_highres, size=secret_half.shape[2:], mode='bilinear',
                                     align_corners=False)
        x = torch.cat([secret_half, container_ds], dim=1)

        shallow = self.shallow_feat(x)
        local_features = self.rdb_blocks(shallow)
        fused = self.global_fusion(local_features)
        coarse_out = self.output_conv(fused + shallow)

        if self.refine:
            return self.refiner(coarse_out)
        else:
            return coarse_out


class GuidedRDN128(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_features=64, num_blocks=6,
                 growth_channels=32, num_layers=5, refine=True):
        super().__init__()
        self.refine = refine
        self.target_size = (128, 128)

        self.shallow_feat = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.rdb_blocks = nn.Sequential(*[
            ResidualDenseBlock(num_features, growth_channels, num_layers)
            for _ in range(num_blocks)
        ])
        self.global_fusion = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

        if refine:
            self.refiner = ResidualRefiner(in_channels=out_channels)

    def forward(self, secret_half, container_highres):
        container_ds = F.interpolate(container_highres, size=secret_half.shape[2:], mode='bilinear',
                                     align_corners=False)
        x = torch.cat([secret_half, container_ds], dim=1)

        shallow = self.shallow_feat(x)
        local_features = self.rdb_blocks(shallow)
        fused = self.global_fusion(local_features)
        coarse_out = self.output_conv(fused + shallow)

        if self.refine:
            out = self.refiner(coarse_out)
        else:
            out = coarse_out

        # Upsample to 128×128
        out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
        return out


class ResidualRefiner(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.refine_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1)
        )
        self.scale = nn.Parameter(torch.tensor(0.1))  # Learnable scale factor

    def forward(self, coarse):
        residual = self.refine_block(coarse)
        return coarse + self.scale * residual


class RDN(nn.Module):
    def __init__(self, in_shape, num_features=64, num_blocks=6, growth_channels=32, num_layers=5):
        super().__init__()
        in_channels, H, W = in_shape
        self.shallow_feat = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.rdb_blocks = nn.Sequential(*[
            ResidualDenseBlock(num_features, growth_channels, num_layers) for _ in range(num_blocks)
        ])

        self.global_fusion = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        shallow = self.shallow_feat(x)
        local_features = self.rdb_blocks(shallow)
        fused = self.global_fusion(local_features)
        out = self.output_conv(fused + shallow)  # global residual
        return out


class GuidedRDN(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_features=64, num_blocks=6,
                 growth_channels=32, num_layers=5, refine=True):
        super().__init__()
        self.refine = refine
        self.shallow_feat = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.rdb_blocks = nn.Sequential(*[
            ResidualDenseBlock(num_features, growth_channels, num_layers)
            for _ in range(num_blocks)
        ])

        self.global_fusion = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

        # Optional MSE-based refinement head
        if refine:
            self.refiner = nn.Sequential(
                nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels, kernel_size=1)
            )

    def forward(self, secret_half, container_highres):
        container_ds = F.interpolate(container_highres, size=secret_half.shape[2:], mode='bilinear',
                                     align_corners=False)
        x = torch.cat([secret_half, container_ds], dim=1)  # [B, 6, 64, 64]

        shallow = self.shallow_feat(x)
        local_features = self.rdb_blocks(shallow)
        fused = self.global_fusion(local_features)
        coarse_out = self.output_conv(fused + shallow)  # Global residual

        if self.refine:
            residual = self.refiner(coarse_out)
            return coarse_out + residual  # Fine correction for PSNR
        else:
            return coarse_out


class UNet_small(nn.Module):
    def __init__(self, in_shape):
        super(UNet_small, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(3, 24, kernel_size=3)  # 512

        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)

        self.center = nn.Sequential(
            ConvBnRelu2d(24, 24, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        out = self.center(out)
        out = self.up1(out, down1)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out


# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Discriminator_Cycle(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator_Cycle, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)  # x, y = 64, depth 64
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)  # (32, 32, 128)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)  # (16, 16, 256)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)  # (8, 8, 512)

        # Classification layer
        self.conv5 = conv(conv_dim * 8, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        # last, classification layer
        out = self.conv5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 [3x256x256]
            nn.Conv2d(in_channels=nc, out_channels=ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 [256x128x128]
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 [512x64x64]
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8 [1024x32x32]
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4 
            nn.Conv2d(in_channels=ndf * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator_WGAN(nn.Module):
    def __init__(self, DIM=64):
        self.DIM = DIM
        super(Discriminator_WGAN, self).__init__()
        self.main = nn.Sequential(
            # input = 3, output = 64
            nn.Conv2d(in_channels=3, out_channels=DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            # input = 64, output = 128
            nn.Conv2d(in_channels=DIM, out_channels=2 * DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            # input = 128, output = 256
            nn.Conv2d(in_channels=2 * DIM, out_channels=4 * DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 4 * self.DIM)
        output = self.linear(output)
        return output


class PatchDiscriminatorHinge(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(PatchDiscriminatorHinge, self).__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Output patch-level real/fake logits
            spectral_norm(nn.Conv2d(dim * 8, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.model(x)  # Shape: (B, 1, H', W')


class Discriminator_WGAN_Two(nn.Module):
    def __init__(self, DIM=64):
        super(Discriminator_WGAN_Two, self).__init__()
        self.DIM = DIM
        self.main = nn.Sequential(
            # Input: (3, 128, 128) - RGB Image
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=DIM, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Input: (DIM, 64, 64)
            spectral_norm(nn.Conv2d(in_channels=DIM, out_channels=2 * DIM, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(2 * DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Input: (2 * DIM, 32, 32)
            spectral_norm(nn.Conv2d(in_channels=2 * DIM, out_channels=4 * DIM, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(4 * DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Input: (4 * DIM, 16, 16)
            spectral_norm(nn.Conv2d(in_channels=4 * DIM, out_channels=8 * DIM, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(8 * DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        # Global Average Pooling and Linear Layer for final output
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(8 * DIM, 1)

    def forward(self, input):
        # Forward pass through convolutional layers
        output = self.main(input)
        # Global Average Pooling
        output = self.gap(output)
        # Flatten the tensor
        output = output.view(-1, 8 * self.DIM)
        # Final Linear Layer
        output = self.linear(output)
        return output


class Discriminator_Cycle_WGAN(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator_Cycle_WGAN, self).__init__()

        # PatchGAN-style discriminator with spectral norm, no sigmoid
        self.conv1 = spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))  # 128 → 64
        self.conv2 = spectral_norm(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1))  # 64 → 32
        self.conv3 = spectral_norm(nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1))  # 32 → 16
        self.conv4 = spectral_norm(nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1))  # 16 → 8
        self.conv5 = spectral_norm(nn.Conv2d(conv_dim * 8, 1, kernel_size=4, stride=1, padding=0))  # 8 → 5

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))  # (B, 64, 64, 64)
        out = self.lrelu(self.conv2(out))  # (B, 128, 32, 32)
        out = self.lrelu(self.conv3(out))  # (B, 256, 16, 16)
        out = self.lrelu(self.conv4(out))  # (B, 512, 8, 8)
        out = self.conv5(out)  # (B, 1, 5, 5) — patch-based output
        return out  # Note: DO NOT apply sigmoid!
