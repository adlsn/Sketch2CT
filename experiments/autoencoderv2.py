import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pathlib
import matplotlib.pyplot as plt
import SimpleITK as sitk
import lpips
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Spacingd, LoadImaged, SaveImaged, ScaleIntensityRange
from monai.data import MetaTensor
from monai.losses import SSIMLoss
from monai.networks.blocks import ResidualUnit
import monai
from tqdm import tqdm

# =============Config==============
device = torch.device("cuda:0")
IMAGE_ROOT = "../data/imagesTr"
LABEL_ROOT = "../data/labelsTr"
BATCH_SIZE = 7
IMAGE_SIZE = 256
TRAIN_STEPS = 1000
SAVE_EVERY = 20
RESULTS_FOLDER = "./autoencoderv2_weight"
INFER_SAVE_DIR = "./autoencoderv2_results"
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx])
        label = sitk.ReadImage(self.label_paths[idx])
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        label_array = sitk.GetArrayFromImage(label).astype(np.float32)

        image_x, image_y, image_z = image_array.shape
        if image_x != IMAGE_SIZE or image_y != IMAGE_SIZE or image_z != IMAGE_SIZE:
            image_array = self.resample(
                image, target_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
            )
        else:
            image_array = torch.tensor(image_array).unsqueeze(0)
            label_array = torch.tensor(label_array).unsqueeze(0)

        transform = ScaleIntensityRange(
            a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        )
        image_array = transform(image_array)
        return image_array

    def resample(self, image, target_shape=(256, 256, 256)):
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
        return strict_resampled_tensor

    @staticmethod
    def create_affine(spacing, origin):
        affine = np.eye(4)
        affine[:3, :3] = np.diag(spacing)
        affine[:3, 3] = origin
        return affine

# =============Autoencoder==============
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
        )
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = out + residual
        return self.relu(out)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(), ResidualBlock(4, 4)
        )
        self.down2 = nn.Sequential(nn.MaxPool3d(2), ResidualBlock(4, 8))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), ResidualBlock(8, 16))
        self.down4 = nn.Sequential(nn.MaxPool3d(2), ResidualBlock(16, 32))

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x4, [x1, x2, x3]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec3 = ResidualBlock(16, 16)
        self.up2 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.dec2 = ResidualBlock(8, 8)
        self.up1 = nn.ConvTranspose3d(8, 4, 2, stride=2)
        self.dec1 = ResidualBlock(4, 4)
        self.final = nn.Conv3d(4, 1, 1)

    def forward(self, x):
        x = self.up3(x)
        # x = self.dec3(x)
        x = self.up2(x)
        # x = self.dec2(x)
        x = self.up1(x)
        # x = self.dec1(x)
        return torch.sigmoid(self.final(x))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent, _ = self.encoder(x)
        return self.decoder(latent)
    

from monai.networks.blocks.dynunet_block import get_conv_layer, UnetBasicBlock
from monai.utils import ensure_tuple_rep, look_up_option
from collections.abc import Sequence
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


# =============Loss==============
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(spatial_dims=3)
        self.lpips = lpips.LPIPS(net='alex').to(device)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        mid = pred.shape[2] // 2
        pred_slice = pred[:, :, mid].repeat(1, 3, 1, 1)  # to 3-ch
        gt_slice = gt[:, :, mid].repeat(1, 3, 1, 1)
        perceptual = self.lpips(pred_slice, gt_slice).mean()
        return self.mse(pred, gt) + self.alpha * self.ssim(pred, gt) + self.beta * perceptual


# =============Inference Visualization==============
def inference_and_save(model, dataset, step, num_samples=2):
    model.eval()
    with torch.no_grad():
        indices = random.sample(range(len(dataset)), num_samples)
        for i, idx in enumerate(indices):
            image = dataset[idx].unsqueeze(0).to(device)
            recon = model(image)
            img_np = image[0, 0].cpu().numpy()
            rec_np = recon[0, 0].cpu().numpy()
            mid = img_np.shape[0] // 2

            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(img_np[mid] * 255, cmap="gray")
            ax[0].set_title("Input")
            ax[1].imshow(rec_np[mid] * 255, cmap="gray")
            ax[1].set_title("Reconstructed")
            for a in ax:
                a.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(INFER_SAVE_DIR, f"step{step}_sample{i}.png"))
            plt.close()
    model.train()


# =============Training==============
def train_autoencoder():
    dataset = NiiDataset(IMAGE_ROOT, LABEL_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model=SwinUNETRAutoencoder(
        img_size=(256, 256, 256),
        in_channels=1,
        out_channels=1,
        feature_size=4,
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 4, 8),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
    ).to(device)

    # model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = CombinedLoss().to(device)

    step = 0
    while step < TRAIN_STEPS:
        displayer = tqdm(dataloader, desc=f"Training Step {step}", unit="step")
        for images in displayer:
            if step >= TRAIN_STEPS:
                break
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            displayer.set_postfix(loss=f"{loss.item():.5f}")

        step += 1

        if step % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"{RESULTS_FOLDER}/autoencoder_step_{step}.pth")
            inference_and_save(model, dataset, step)


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(INFER_SAVE_DIR, exist_ok=True)
    train_autoencoder()


if __name__ == "__main__":
    main()
