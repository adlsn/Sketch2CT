import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import (
    SwinTransformer,
    PatchMergingV2,
    PatchMerging,
    MERGING_MODE,
)
from monai.networks.blocks import (
    PatchEmbed,
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrUpBlock,
)

from monai.utils import ensure_tuple_rep, look_up_option
import numpy as np
import sys
import pathlib
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import SimpleITK as sitk

sys.path.append("..")
sys.path.insert(1, "./")
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.transforms import Spacingd, LoadImaged, SaveImaged
from monai.data import MetaTensor
from monai.transforms import ScaleIntensityRange
from monai.networks.nets import DynUNet
from monai.networks.layers import Norm
from monai.networks.nets import UNETR
from monai.losses import SSIMLoss
from monai.networks.nets import SwinUNETR, SegResNet
from monai.networks.blocks import SABlock
from collections.abc import Sequence
from monai.networks.blocks.dynunet_block import get_conv_layer, UnetBasicBlock

class RecoverFreq:
    def __init__(
        self, full_size: int = 256, block_size: int = 64, corner: str = "corner"
    ):
        """
        用于在频域 volume 中提取与还原高频块的类。

        参数:
            full_size: 原始频域 volume 的大小(默认 256)
            block_size: 高频 crop 的大小(默认 64)
            corner: 提取/插入位置，可选 'corner' 或 'random'
        """
        self.full_size = full_size
        self.block_size = block_size
        self.corner = corner
        self.random_start = None  # 仅用于 random 模式时记录位置

    def compress(self, hf_volume: torch.Tensor) -> torch.Tensor:
        """
        从频域 volume 中提取高频区域(默认从 corner)。

        返回:
            hf_crop: torch.Tensor, [block_size, block_size, block_size]
        """
        D, H, W = hf_volume.shape
        assert D == H == W == self.full_size, "Volume size mismatch."

        if self.corner == "corner":
            return hf_volume[: self.block_size, : self.block_size, : self.block_size]
        elif self.corner == "random":
            self.random_start = [
                torch.randint(0, D - self.block_size, (1,)).item(),
                torch.randint(0, H - self.block_size, (1,)).item(),
                torch.randint(0, W - self.block_size, (1,)).item(),
            ]
            s = self.random_start
            return hf_volume[
                s[0] : s[0] + self.block_size,
                s[1] : s[1] + self.block_size,
                s[2] : s[2] + self.block_size,
            ]
        else:
            raise NotImplementedError("Only 'corner' or 'random' mode supported.")

    def recover(self, hf_crop: torch.Tensor) -> torch.Tensor:
        """
        将提取出的高频块还原到完整频域 volume 的对应位置。

        返回:
            hf_volume: torch.Tensor, [full_size, full_size, full_size]
        """
        hf_volume = torch.zeros(
            (self.full_size, self.full_size, self.full_size),
            dtype=hf_crop.dtype,
            device=hf_crop.device,
        )

        if self.corner == "corner":
            hf_volume[: self.block_size, : self.block_size, : self.block_size] = hf_crop
        elif self.corner == "random":
            if self.random_start is None:
                raise ValueError("Must call compress() first to store random position.")
            s = self.random_start
            hf_volume[
                s[0] : s[0] + self.block_size,
                s[1] : s[1] + self.block_size,
                s[2] : s[2] + self.block_size,
            ] = hf_crop
        else:
            raise NotImplementedError("Only 'corner' or 'random' mode supported.")

        return hf_volume


class HFreq:
    """
    处理复数高频 volume 的类, 支持从复数转换为 (幅值, 相位), 以及反向恢复。
    """

    @staticmethod
    def complex_to_mag_phase(complex_volume: torch.Tensor) -> torch.Tensor:
        """
        将复数 volume 分解为幅值和相位。

        Args:
            complex_volume (torch.Tensor): 形状为 [D, H, W] 或 [B, D, H, W] 的复数张量 (dtype=torch.complex64)

        Returns:
            torch.Tensor: 形状为 [2, D, H, W] 或 [B, 2, D, H, W], channel=0 是幅值, channel=1 是相位
        """
        magnitude = torch.abs(complex_volume)
        phase = torch.angle(complex_volume)
        return torch.stack(
            [magnitude, phase], dim=-4 if complex_volume.ndim == 3 else 1
        )

    @staticmethod
    def mag_phase_to_complex(mag_phase_volume: torch.Tensor) -> torch.Tensor:
        """
        将幅值和相位张量恢复为复数张量。

        Args:
            mag_phase_volume (torch.Tensor): 形状为 [2, D, H, W] 或 [B, 2, D, H, W] 的张量

        Returns:
            torch.Tensor: 形状为 [D, H, W] 或 [B, D, H, W] 的复数张量
        """
        if mag_phase_volume.ndim == 4:
            magnitude = mag_phase_volume[0]
            phase = mag_phase_volume[1]
        elif mag_phase_volume.ndim == 5:
            magnitude = mag_phase_volume[:, 0]
            phase = mag_phase_volume[:, 1]
        else:
            raise ValueError(
                "Expected input shape to be [2, D, H, W] or [B, 2, D, H, W]"
            )

        return magnitude * torch.exp(1j * phase)


def extract_high_low(volume: torch.Tensor, high_ratio=0.25):
    fft = torch.fft.fftn(volume, dim=(-3, -2, -1))
    fft_shifted = torch.fft.fftshift(fft)

    D, H, W = volume.shape
    center_d, center_h, center_w = D // 2, H // 2, W // 2
    margin_d = int(D * (1 - high_ratio) / 2)
    margin_h = int(H * (1 - high_ratio) / 2)
    margin_w = int(W * (1 - high_ratio) / 2)

    mask = torch.ones_like(fft_shifted)
    mask[
        center_d - margin_d : center_d + margin_d,
        center_h - margin_h : center_h + margin_h,
        center_w - margin_w : center_w + margin_w,
    ] = 0

    high_freq = fft_shifted * mask
    low_freq = fft_shifted * (1 - mask)
    recover = RecoverFreq()
    high_freq = recover.compress(high_freq)
    return high_freq, low_freq


def downsample(volume, target_size=(64, 64, 64)):
    # volume: Tensor [D, H, W]
    D, H, W = volume.shape
    volume_ds = F.interpolate(
        volume[None, None], size=target_size, mode="trilinear", align_corners=False
    )
    return volume_ds.squeeze()


def upsample(volume, target_size=(256, 256, 256)):
    volume_us = F.interpolate(
        volume[None, None], size=target_size, mode="trilinear", align_corners=False
    )
    return volume_us.squeeze()


def reconstruct_from_high_low(
    upsampled: torch.Tensor, high_freq: torch.Tensor, high_ratio=0.25
):
    high_freq_upsampled, low_freq = extract_high_low(upsampled, high_ratio=high_ratio)
    recover = RecoverFreq()
    high_freq_upsampled = recover.recover(high_freq_upsampled)  # [256, 256, 256]
    full_freq = high_freq + low_freq + high_freq_upsampled
    full_freq_ishift = torch.fft.ifftshift(full_freq)
    reconstructed = torch.fft.ifftn(full_freq_ishift, dim=(-3, -2, -1)).real
    return reconstructed


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(spatial_dims=3)
        self.alpha = alpha

    def forward(self, pred, gt):
        return self.mse(pred, gt) + self.alpha * self.ssim(pred, gt)