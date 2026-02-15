"""
TrackNet - Ball Tracking Neural Network
U-Net style encoder-decoder architecture for ball detection in sports videos.

Based on: "TrackNet: A Deep Learning Network for Tracking High-speed
and Tiny Objects in Sports Applications" (Huang et al.)

Architecture matches padel_analytics implementation for weight compatibility.
Input: 9 consecutive RGB frames (27 channels)
Output: 8-channel heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DBlock(nn.Module):
    """Basic convolution block: Conv2D (no bias) -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Double2DConv(nn.Module):
    """Two consecutive convolution blocks - named conv_1, conv_2 for weight compatibility"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_channels, out_channels)
        self.conv_2 = Conv2DBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Triple2DConv(nn.Module):
    """Three consecutive convolution blocks - named conv_1, conv_2, conv_3 for weight compatibility"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_channels, out_channels)
        self.conv_2 = Conv2DBlock(out_channels, out_channels)
        self.conv_3 = Conv2DBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class TrackNet(nn.Module):
    """
    TrackNet: U-Net style encoder-decoder for ball tracking.

    Takes 9 consecutive frames as input (27 channels: 9 frames * 3 RGB)
    Outputs an 8-channel heatmap (for ball position probability).

    Architecture matches padel_analytics for weight compatibility:
    - Encoder: 3 down-sampling blocks (64 -> 128 -> 256 -> 512)
    - Decoder: 3 up-sampling blocks with skip connections
    - Output: 8 channels (predictor layer)
    """

    def __init__(self, in_dim=27, out_dim=8):
        """
        Args:
            in_dim: Number of input channels (default 27 for 9 RGB frames)
            out_dim: Number of output channels (default 8 for heatmap)
        """
        super().__init__()

        # Encoder (downsampling path)
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)

        # Bottleneck
        self.bottleneck = Triple2DConv(256, 512)

        # Decoder (upsampling path with skip connections)
        self.up_block_1 = Triple2DConv(768, 256)   # 512 + 256
        self.up_block_2 = Double2DConv(384, 128)   # 256 + 128
        self.up_block_3 = Double2DConv(192, 64)    # 128 + 64

        # Output layer - named 'predictor' for weight compatibility
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.down_block_1(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)

        x2 = self.down_block_2(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)

        x3 = self.down_block_3(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)

        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)

        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)

        # Output (apply sigmoid for independent probabilities)
        x = self.predictor(x)
        x = self.sigmoid(x)

        return x


class Conv1DBlock(nn.Module):
    """1D convolution block for InpaintNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')

    def forward(self, x):
        return F.relu(self.conv(x))


class Bottleneck1D(nn.Module):
    """Bottleneck with named conv layers for weight compatibility"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = Conv1DBlock(in_ch, out_ch)
        self.conv_2 = Conv1DBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class InpaintNet(nn.Module):
    """
    InpaintNet: 1D U-Net for trajectory inpainting.

    Takes ball trajectory coordinates and fills in missing positions.
    Input: (batch, 3, seq_len) where 3 = (x, y, visibility)
    Output: (batch, 2, seq_len) where 2 = (x, y) predicted
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)

        # Bottleneck (with named layers to match weights)
        self.buttleneck = Bottleneck1D(128, 256)

        # Decoder with skip connections
        self.up_1 = Conv1DBlock(256 + 128, 128)
        self.up_2 = Conv1DBlock(128 + 64, 64)
        self.up_3 = Conv1DBlock(64 + 32, 32)

        # Output
        self.predictor = nn.Conv1d(32, 2, kernel_size=3, padding='same')

    def forward(self, x):
        # Encoder
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        # Bottleneck
        x = self.buttleneck(x3)

        # Decoder with skip connections
        x = torch.cat([x, x3], dim=1)
        x = self.up_1(x)

        x = torch.cat([x, x2], dim=1)
        x = self.up_2(x)

        x = torch.cat([x, x1], dim=1)
        x = self.up_3(x)

        # Output
        x = self.predictor(x)
        return x


def load_inpaintnet(weights_path, device='cpu'):
    """Load InpaintNet model with pretrained weights."""
    model = InpaintNet()

    checkpoint = torch.load(weights_path, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


def load_tracknet(weights_path, device='cpu'):
    """
    Load TrackNet model with pretrained weights.

    Args:
        weights_path: Path to the .pt weights file
        device: 'cpu' or 'cuda'

    Returns:
        TrackNet model with loaded weights
    """
    model = TrackNet(in_dim=27, out_dim=8)

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model
