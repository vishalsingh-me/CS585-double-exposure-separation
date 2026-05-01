import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DualHeadUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*2, base_channels*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*4, base_channels*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*8, base_channels*16))
        
        self.up1 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels*16, base_channels*8)
        
        self.up2 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels*8, base_channels*4)
        
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels*4, base_channels*2)
        
        self.up4 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels*2, base_channels)
        
        self.outc = nn.Conv2d(base_channels, 6, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        u1 = self.up1(x5)
        # Pad if necessary
        u1 = torch.cat([u1, x4], dim=1)
        c1 = self.conv1(u1)
        
        u2 = self.up2(c1)
        u2 = torch.cat([u2, x3], dim=1)
        c2 = self.conv2(u2)
        
        u3 = self.up3(c2)
        u3 = torch.cat([u3, x2], dim=1)
        c3 = self.conv3(u3)
        
        u4 = self.up4(c3)
        u4 = torch.cat([u4, x1], dim=1)
        c4 = self.conv4(u4)
        
        out = self.outc(c4)
        out = self.sigmoid(out)  # [0, 1]
        
        pred_1 = out[:, 0:3, :, :]
        pred_2 = out[:, 3:6, :, :]
        
        return pred_1, pred_2
