import torch
import torch.nn as nn
from src.models.two_decoder_unet import DoubleConv, Decoder

class RefinementUNet(nn.Module):
    """Lightweight U-Net for residual refinement."""
    def __init__(self, in_channels=9, out_channels=6, base_channels=16):
        super().__init__()
        c = base_channels
        
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(c * 4, c * 8)
        
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        
        self.out = nn.Conv2d(c, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

class TwoStageRefinementUNet(nn.Module):
    """
    Two-Stage Residual Refinement Separator for double-exposure separation.
    Stage 1: Shared encoder, two decoders to generate coarse predictions.
    Stage 2: Lightweight refiner to generate residuals.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        c = base_channels
        
        # Stage 1: Coarse prediction (similar to TwoDecoderUNet)
        self.block1 = DoubleConv(in_channels, c)
        self.block2 = DoubleConv(c, c * 2)
        self.block3 = DoubleConv(c * 2, c * 4)
        self.block4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)
        
        self.decoder_1 = Decoder(c)
        self.decoder_2 = Decoder(c)
        
        # Stage 2: Refinement
        # Use half the base channels for the refiner to keep it lightweight, min 16
        refiner_base = max(16, c // 2)
        self.refiner = RefinementUNet(in_channels=9, out_channels=6, base_channels=refiner_base)

    def forward(self, x):
        # Stage 1: Coarse predictions
        x1 = self.block1(x)
        x2 = self.block2(self.pool(x1))
        x3 = self.block3(self.pool(x2))
        x4 = self.block4(self.pool(x3))
        bottleneck = self.bottleneck(self.pool(x4))
        
        skips = (x1, x2, x3, x4)
        coarse_1 = self.decoder_1(bottleneck, skips)
        coarse_2 = self.decoder_2(bottleneck, skips)
        
        # Stage 2: Refinement
        # Input to refiner is concatenated mixture and coarse predictions
        refiner_in = torch.cat([x, coarse_1, coarse_2], dim=1)
        residual = self.refiner(refiner_in)
        
        # Residual correction
        residual_1 = residual[:, 0:3]
        residual_2 = residual[:, 3:6]
        
        pred_1 = torch.clamp(coarse_1 + residual_1, 0.0, 1.0)
        pred_2 = torch.clamp(coarse_2 + residual_2, 0.0, 1.0)
        
        return pred_1, pred_2
