import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        c = base_channels
        self.up1 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(c * 16, c * 8)
        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(c * 8, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(c * 4, c * 2)
        self.up4 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, 3, kernel_size=1)

    def forward(self, bottleneck, skips):
        x1, x2, x3, x4 = skips

        y = self.up1(bottleneck)
        y = torch.cat([y, x4], dim=1)
        y = self.conv1(y)

        y = self.up2(y)
        y = torch.cat([y, x3], dim=1)
        y = self.conv2(y)

        y = self.up3(y)
        y = torch.cat([y, x2], dim=1)
        y = self.conv3(y)

        y = self.up4(y)
        y = torch.cat([y, x1], dim=1)
        y = self.conv4(y)

        return torch.sigmoid(self.out(y))


class TwoDecoderUNet(nn.Module):
    """
    Shared-encoder U-Net with independent decoders for the two source layers.
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        c = base_channels

        self.block1 = DoubleConv(in_channels, c)
        self.block2 = DoubleConv(c, c * 2)
        self.block3 = DoubleConv(c * 2, c * 4)
        self.block4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)

        self.decoder_1 = Decoder(c)
        self.decoder_2 = Decoder(c)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(self.pool(x1))
        x3 = self.block3(self.pool(x2))
        x4 = self.block4(self.pool(x3))
        bottleneck = self.bottleneck(self.pool(x4))

        skips = (x1, x2, x3, x4)
        pred_1 = self.decoder_1(bottleneck, skips)
        pred_2 = self.decoder_2(bottleneck, skips)
        return pred_1, pred_2
