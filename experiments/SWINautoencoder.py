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
from accelerate import Accelerator

# =============Config==============
device = torch.device("cuda:0")
IMAGE_ROOT = "../data/imagesTr"
LABEL_ROOT = "../data/labelsTr"
BATCH_SIZE = 16
IMAGE_SIZE = 256
TRAIN_STEPS = 1000
SAVE_EVERY = 10
RESULTS_FOLDER = "./swinautoencoder_weight"
INFER_SAVE_DIR = "./swinautoencoder_results"

# MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}
# =================================

torch.random.manual_seed(66)
np.random.seed(66)


# =============Dataset==============
class NiiDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        image_dir = pathlib.Path(image_dir)
        label_dir = pathlib.Path(label_dir)
        self.image_paths = sorted([str(f) for f in list(image_dir.rglob("*.nii.gz"))])
        self.label_paths = sorted([str(f) for f in list(label_dir.rglob("*.nii.gz"))])
        print(f"{len(self.image_paths)} volumes found in {image_dir}")

        # Preprocess
        self.image_lists = [sitk.ReadImage(path) for path in self.image_paths]
        self.resampled_images = []
        for image in self.image_lists:
            image_array = sitk.GetArrayFromImage(image).astype(np.float32)
            image_x, image_y, image_z = image_array.shape
            transform = ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
            )
            if image_x != IMAGE_SIZE or image_y != IMAGE_SIZE or image_z != IMAGE_SIZE:
                resampled_array = self.resample(
                    image, target_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
                )
                resampled_array = transform(resampled_array)
                self.resampled_images.append(resampled_array)
            else:
                image_array = transform(image_array.unsqueeze(0))
                self.resampled_images.append(torch.tensor(image_array))
        print(f"Preprocessed {len(self.resampled_images)} volumes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = sitk.ReadImage(self.image_paths[idx])
        # label = sitk.ReadImage(self.label_paths[idx])
        # image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        # label_array = sitk.GetArrayFromImage(label).astype(np.float32)

        # image_x, image_y, image_z = image_array.shape
        # if image_x != IMAGE_SIZE or image_y != IMAGE_SIZE or image_z != IMAGE_SIZE:
        #     image_array = self.resample(
        #         image, target_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        #     )
        # else:

        #     image_array = torch.tensor(image_array).unsqueeze(0)  # [1, D, H, W]
        #     label_array = torch.tensor(label_array).unsqueeze(0)  # [1, D, H, W]

        # transform = ScaleIntensityRange(
        #     a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        # )
        # image_array = transform(image_array)
        # # return image_array, label_array  # Return only segmentation as target
        # return image_array  # Return only medical volume
        return self.resampled_images[idx]  # Return preprocessed image

    def resample(self, image, target_shape=(512, 512, 512)):
        origin = image.GetOrigin()
        old_spacing = image.GetSpacing()
        image_array = sitk.GetArrayFromImage(image)
        original_spacing = list(old_spacing)
        old_shape = image_array.shape
        new_spacing = [
            original_spacing[i] * old_shape[i] / target_shape[i] for i in range(3)
        ]

        affine_matrix = self.create_affine(original_spacing, origin)
        image_tensor = torch.tensor(image_array).unsqueeze(0)
        meta_tensor = MetaTensor(image_tensor, affine_matrix)
        resampler = monai.transforms.Spacing(pixdim=new_spacing, mode="bilinear")
        resampled_tensor = resampler(meta_tensor)

        resample_transform = monai.transforms.ResizeWithPadOrCrop(
            spatial_size=target_shape
        )
        strict_resampled_tensor = resample_transform(resampled_tensor)

        resampled_array = strict_resampled_tensor
        return resampled_array  # [1, D, H, W]

    @staticmethod
    def create_affine(spacing, origin):
        affine = np.eye(4)  # 创建一个 4x4 单位矩阵
        affine[:3, :3] = np.diag(spacing)  # 在对角线上填入 spacing
        affine[:3, 3] = origin  # 填入 origin 信息
        return affine


class UnetrUpBlockv2(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = (
            "leakyrelu",
            {"inplace": True, "negative_slope": 0.01},
        ),
        dropout: tuple | str | float | None = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp):
        out = self.transp_conv(inp)
        out = self.conv_block(out)
        return out


class SwinUNETRAutoencoder(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        feature_size=8,
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 6, 8),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
    ):
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        # --- Encoder ---
        self.encoder["swin"] = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE),
            use_v2=use_v2,
        )

        self.encoder["stage0"] = UnetrBasicBlock(
            spatial_dims, in_channels, feature_size, 3, 1, norm_name, res_block=True
        )
        self.encoder["stage4"] = UnetrBasicBlock(
            spatial_dims,
            16 * feature_size,
            16 * feature_size,
            3,
            1,
            norm_name,
            res_block=True,
        )

        # --- Decoder without skip connections ---
        self.decoder["up3"] = UnetrUpBlockv2(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,  # 384
            out_channels=8 * feature_size,  # 192
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            stride=1,
        )
        self.decoder["up2"] = UnetrUpBlockv2(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,  # 192
            out_channels=4 * feature_size,  # 96
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            stride=1,
        )
        self.decoder["up1"] = UnetrUpBlockv2(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,  # 96
            out_channels=2 * feature_size,  # 48
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            stride=1,
        )
        self.decoder["up0"] = UnetrUpBlockv2(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,  # 48
            out_channels=feature_size,  # 24
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            stride=1,
        )

        self.decoder["up_final"] = UnetrUpBlockv2(
            spatial_dims=spatial_dims,
            in_channels=feature_size,  # 24
            out_channels=feature_size,  # 24 (or reduce to 12 if you want)
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            stride=1,
        )

        self.decoder["out"] = UnetOutBlock(spatial_dims, feature_size, out_channels)

    def forward_encoder(self, x):
        x0 = self.encoder["stage0"](x)
        hidden_states = self.encoder["swin"](x)
        x4 = self.encoder["stage4"](hidden_states[4])  # bottleneck latent
        return x4  # only return latent

    def forward_decoder(self, x):
        x = self.decoder["up3"](x)
        x = self.decoder["up2"](x)
        x = self.decoder["up1"](x)
        x = self.decoder["up0"](x)
        x = self.decoder["up_final"](x)
        return self.decoder["out"](x)

    def forward(self, x):
        latent = self.forward_encoder(x)
        return self.forward_decoder(latent)


# x = torch.randn(1, 1, 64, 64, 64).cuda()  # Example input tensor
# model = SwinUNETRAutoencoder(
#     img_size=(64, 64, 64),
#     in_channels=1,
#     out_channels=1,
#     feature_size=24,
#     depths=(2, 2, 2, 2),
#     num_heads=(3, 6, 12, 24),
#     norm_name="instance",
#     drop_rate=0.0,
#     attn_drop_rate=0.0,
#     dropout_path_rate=0.0,
#     normalize=True,
#     use_checkpoint=False,
#     spatial_dims=3,
#     downsample="merging",
#     use_v2=False,
# ).cuda()
# output = model(x)
# print(output.shape)  # Should print the shape of the output tensor
# o1 = model.forward_encoder(x)
# print(o1.shape)  # Should print the shape of the latent tensor
# o2 = model.forward_decoder(o1)
# print(o2.shape)  # Should print the shape of the output tensor after decoding


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


def inference_and_save(ae_high, ae_low, dataset, step, num_samples=2):
    ae_high.eval()
    ae_low.eval()
    with torch.no_grad():
        indices = random.sample(range(len(dataset)), num_samples)
        for i, idx in enumerate(indices):
            image = dataset[idx].unsqueeze(0).to(device)  # [1,1,D,H,W]
            high_freq, low_freq = extract_high_low(image.squeeze(), high_ratio=0.25)
            low_freq = downsample(image.squeeze(), target_size=(64, 64, 64))
            low_tensor = low_freq
            recover = RecoverFreq()
            # high_freq_resample = recover.compress(high_freq)
            hf_process_resample = HFreq.complex_to_mag_phase(high_freq)  # 2*64*64*64

            pred_high = ae_high(hf_process_resample.unsqueeze(0)).squeeze()
            pred_low = ae_low(low_tensor.unsqueeze(0).unsqueeze(0)).squeeze()
            pred_low = upsample(pred_low, target_size=(256, 256, 256))

            pred_high = HFreq.mag_phase_to_complex(pred_high)
            pred_high = recover.recover(pred_high)
            reconstructed = reconstruct_from_high_low(pred_low, pred_high)

            # reconstructed = (reconstructed + 1.0) / 2.0 * 255.0  # [0, 255]
            reconstructed = reconstructed.cpu().numpy()
            image_np = image.squeeze().cpu().numpy()
            mid = image_np.shape[0] // 2
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image_np[mid] * 255, cmap="gray")
            plt.title("Original Image")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed[mid] * 255, cmap="gray")
            plt.title("Reconstructed Image")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"{INFER_SAVE_DIR}/sample_{i}_step_{step}.png")
            plt.close()


def train_autoencoder(checkpoint=False):
    dataset = NiiDataset(IMAGE_ROOT, LABEL_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    ae_high = SwinUNETRAutoencoder(
        img_size=(64, 64, 64),
        in_channels=2,
        out_channels=2,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
    )
    ae_low = SwinUNETRAutoencoder(
        img_size=(64, 64, 64),
        in_channels=1,
        out_channels=1,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
    )

    optimizer_high = torch.optim.Adam(ae_high.parameters(), lr=1e-4)
    optimizer_low = torch.optim.Adam(ae_low.parameters(), lr=1e-4)

    criterion = CombinedLoss().to(device)

    accelerator = Accelerator()
    ae_high, ae_low, optimizer_high, optimizer_low, dataloader = accelerator.prepare(
        ae_high, ae_low, optimizer_high, optimizer_low, dataloader
    )

    # load checkpoint
    if checkpoint:
        checkpoint_path = f"{RESULTS_FOLDER}/autoencoder_step_30.pth"  # TODO: change 30 to the milestone step you want to load
        if os.path.exists(checkpoint_path):
            weights = torch.load(checkpoint_path, map_location=device)
            accelerator.unwrap_model(ae_high).load_state_dict(weights["ae_high"])
            accelerator.unwrap_model(ae_low).load_state_dict(weights["ae_low"])
            optimizer_high.load_state_dict(weights["optimizer_high"])
            optimizer_low.load_state_dict(weights["optimizer_low"])
            step = weights["step"]
            print(f"Loaded checkpoint from {checkpoint_path} at step {step}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
            step = 0
    else:
        print("Starting training from scratch.")
        step = 0

    while step < TRAIN_STEPS:
        displayer = tqdm(dataloader, desc=f"Training Step {step}", unit="step")
        for images in displayer:
            if step >= TRAIN_STEPS:
                break
            volume = images.to(device)  # [B, 1, D, H, W] normalized in [-1, 1]
            volume_denorm = (volume + 1.0) / 2.0 * 255.0  # [0, 255]

            outputs_high, outputs_low = [], []

            for b in range(volume.shape[0]):
                vol = volume_denorm[b, 0]  # shape [D, H, W]
                high_freq, _ = extract_high_low(vol, high_ratio=0.25)

                low_freq = downsample(vol, target_size=(64, 64, 64))
                low_tensor = low_freq

                # recover = RecoverFreq()

                # high_freq = recover.compress(high_freq)
                hf_process = HFreq.complex_to_mag_phase(high_freq)
                high_tensor = hf_process
                # high_freq = torch.nan_to_num(high_freq, nan=0.0, posinf=0.0, neginf=0.0)
                # high_tensor = torch.abs(high_freq)

                outputs_high.append(high_tensor)
                outputs_low.append(low_tensor)

            x_high = torch.stack(outputs_high)  # [B, 2, D, H, W]
            x_low = torch.stack(outputs_low).unsqueeze(1)

            x_high = ((x_high - x_high.min()) / (x_high.max() - x_high.min())) * 2 - 1
            x_low = ((x_low - x_low.min()) / (x_low.max() - x_low.min())) * 2 - 1

            pred_high = ae_high(x_high)
            pred_low = ae_low(x_low)

            loss_high = criterion(pred_high, x_high)
            loss_low = criterion(pred_low, x_low)
            total_loss = loss_high + loss_low

            optimizer_high.zero_grad()
            optimizer_low.zero_grad()
            # loss_high.backward()
            # loss_low.backward()
            accelerator.backward(loss_high)
            accelerator.clip_grad_norm_(ae_high.parameters(), max_norm=1.0)
            accelerator.backward(loss_low)
            accelerator.clip_grad_norm_(ae_low.parameters(), max_norm=1.0)
            optimizer_high.step()
            optimizer_low.step()

            displayer.set_postfix(
                loss_high=f"{loss_high.item():.5f}",
                loss_low=f"{loss_low.item():.5f}",
                total_loss=f"{total_loss.item():.5f}",
            )

        step += 1

        if step % SAVE_EVERY == 0:

            weights = {
                "ae_high": accelerator.get_state_dict(ae_high),
                "ae_low": accelerator.get_state_dict(ae_low),
                "optimizer_high": optimizer_high.state_dict(),
                "optimizer_low": optimizer_low.state_dict(),
                "step": step,
            }

            torch.save(weights, f"{RESULTS_FOLDER}/autoencoder_step_{step}.pth")

            # torch.save(ae_high.state_dict(), f"{RESULTS_FOLDER}/high_{step}.pt")
            # torch.save(ae_low.state_dict(), f"{RESULTS_FOLDER}/low_{step}.pt")
            inference_and_save(ae_high, ae_low, dataset, step)


if __name__ == "__main__":
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.exists(INFER_SAVE_DIR):
        os.makedirs(INFER_SAVE_DIR)

    train_autoencoder()
    print("Training completed and results saved.")

    # x = torch.randn(1, 1, 256, 256, 256).to(device)
    # high, low = extract_high_low(x.squeeze(), high_ratio=0.25)
    # print(
    #     high.shape, low.shape
    # )  # Should print the shapes of high and low frequency tensors
