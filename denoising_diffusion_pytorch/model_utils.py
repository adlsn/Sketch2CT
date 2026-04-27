import torch
import numpy as np
import torch.nn as nn
from monai.networks.nets import DynUNet
from monai.networks.blocks import UnetOutBlock
from typing import Optional, Sequence, Union, List, Tuple


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(np.log(10000) / half_dim)
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb  # [B, dim]


class DynUNetWithTimeEmbedding(torch.nn.Module):
    def __init__(self, base_unet: torch.nn.Module, time_emb_dim=16):
        super().__init__()
        self.base_unet = base_unet  # 传入原始 DynUNet
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_emb_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
        )

    def forward(self, x, t):
        B, C, D, H, W = x.shape  # 获取 shape
        temb = self.time_embedding(t)  # shape: [B, time_emb_dim]
        temb = self.time_mlp(temb)  # shape: [B, 512]
        temb = temb[:, :, None, None, None]  # shape: [B, 512, 1, 1, 1]

        temb = temb.expand(-1, -1, D, H, W)  # shape: [B, 512, D, H, W]
        x = torch.cat([x, temb], dim=1)  # 输入变成 [B, 1+512, D, H, W]

        return self.base_unet(x)


def get_out_channels(module):
    for sub in reversed(list(module.modules())):
        if isinstance(sub, (nn.Conv3d, nn.ConvTranspose3d)):
            return sub.out_channels
    raise RuntimeError("Could not find Conv3d in module to determine out_channels")


class SKUNet(DynUNet):
    def __init__(self, time_emb_dim: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_emb_dim = time_emb_dim
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)

        # 1 input + N downs + 1 bottleneck + N ups
        self.total_blocks = 1 + len(self.downsamples) + 1 + len(self.upsamples)
        self.temb_proj = nn.ModuleList()

        # Get each block's output channel count
        block_channels = (
            [get_out_channels(self.input_block)]
            + [get_out_channels(f) for f in self.downsamples]
            + [get_out_channels(self.bottleneck)]
            + [get_out_channels(f) for f in self.upsamples]
        )

        for ch in block_channels:
            self.temb_proj.append(
                nn.Sequential(nn.Linear(time_emb_dim, ch), nn.SiLU(), nn.Linear(ch, ch))
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)  # [B, time_emb_dim]

        # Inject into input block
        x = self.input_block(x)
        x = self._inject_time(x, t_emb, self.temb_proj[0])
        features = [x]

        # Downsampling
        for i, down in enumerate(self.downsamples):
            x = down(x)
            x = self._inject_time(x, t_emb, self.temb_proj[i + 1])
            features.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self._inject_time(x, t_emb, self.temb_proj[len(self.downsamples) + 1])

        # Upsampling
        for i, up in enumerate(self.upsamples):
            skip_x = features[-(i + 1)]
            x = up(x, skip_x)
            x = self._inject_time(
                x, t_emb, self.temb_proj[len(self.downsamples) + 2 + i]
            )

        x = self.output_block(x)
        return x

    def _inject_time(
        self, x: torch.Tensor, t_emb: torch.Tensor, projector: nn.Module
    ) -> torch.Tensor:
        """
        Inject projected time embedding into feature map by additive modulation.
        """
        t_proj = (
            projector(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # [B, C, 1, 1, 1]
        return x + t_proj


if __name__ == "__main__":
    # Example usage
    t = torch.tensor([1, 2, 3])
    pos_emb = SinusoidalPositionEmbeddings(dim=128)
    output = pos_emb(t)
    print(output.shape)  # Should be [3, 128]
