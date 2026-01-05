"""
Generator architectures for adversarial perturbation generation.

Architectures:
- SimpleCNN: Basic encoder-decoder (~500K params)
- UNet: Skip connections (~2M params)
- ResUNet: UNet + residual blocks (~3M params)
- AttentionUNet: UNet + attention gates (~3.5M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g + s)
        psi = self.psi(psi)
        return skip * psi


class SimpleCNN(nn.Module):
    """Simple encoder-decoder without skip connections. ~500K params."""
    
    def __init__(self, base_channels: int = 32, **kwargs):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(base_channels, 3, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d1 = self.dec1(e4)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        return self.final(d3)


class UNet(nn.Module):
    """UNet with skip connections. ~2M params."""
    
    def __init__(self, base_channels: int = 32, **kwargs):
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.final = nn.Conv2d(base_channels, 3, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class ResUNet(nn.Module):
    """UNet with residual blocks. ~3M params."""
    
    def __init__(self, base_channels: int = 32, num_blocks: int = 2, **kwargs):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvBlock(3, base_channels),
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            *[ResidualBlock(base_channels * 2) for _ in range(num_blocks)]
        )
        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            *[ResidualBlock(base_channels * 4) for _ in range(num_blocks)]
        )
        self.enc4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            *[ResidualBlock(base_channels * 8) for _ in range(num_blocks)]
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            *[ResidualBlock(base_channels * 16) for _ in range(num_blocks)]
        )
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = nn.Sequential(ConvBlock(base_channels * 16, base_channels * 8), ResidualBlock(base_channels * 8))
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(base_channels * 8, base_channels * 4), ResidualBlock(base_channels * 4))
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(base_channels * 4, base_channels * 2), ResidualBlock(base_channels * 2))
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(base_channels * 2, base_channels), ResidualBlock(base_channels))
        self.final = nn.Conv2d(base_channels, 3, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class AttentionUNet(nn.Module):
    """UNet with attention gates. ~3.5M params."""
    
    def __init__(self, base_channels: int = 32, **kwargs):
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.att4 = AttentionGate(base_channels * 16, base_channels * 8, base_channels * 4)
        self.att3 = AttentionGate(base_channels * 8, base_channels * 4, base_channels * 2)
        self.att2 = AttentionGate(base_channels * 4, base_channels * 2, base_channels)
        self.att1 = AttentionGate(base_channels * 2, base_channels, base_channels // 2)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.final = nn.Conv2d(base_channels, 3, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        up4 = self.up4(b)
        d4 = self.dec4(torch.cat([up4, self.att4(up4, e4)], dim=1))
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat([up3, self.att3(up3, e3)], dim=1))
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2, self.att2(up2, e2)], dim=1))
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat([up1, self.att1(up1, e1)], dim=1))
        return self.final(d1)


# Factory function
ARCHITECTURES = {
    'SimpleCNN': SimpleCNN,
    'UNet': UNet,
    'ResUNet': ResUNet,
    'AttentionUNet': AttentionUNet
}


def create_mapper(architecture: str, **kwargs) -> nn.Module:
    """Create a mapper by name."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[architecture](**kwargs)



