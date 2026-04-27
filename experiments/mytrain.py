import torch
import numpy as np
import sys
import SimpleITK as sitk
import monai
from monai.transforms import Spacingd, LoadImaged, SaveImaged
from monai.data import MetaTensor
from monai.transforms import ScaleIntensityRange
from monai.networks.nets import DynUNet
from monai.networks.layers import Norm
import random
import matplotlib.pyplot as plt

sys.path.append("..")
sys.path.insert(1, "./")
from torch.utils.data import Dataset, DataLoader
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion
from denoising_diffusion_pytorch.model_utils import DynUNetWithTimeEmbedding, SKUNet
from denoising_diffusion_pytorch.sketch_diffusion import SKViT3D, GaussianDiffusion3D
from denoising_diffusion_pytorch import Trainer3D

# from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D
from torch.optim import Adam
from accelerate import Accelerator
from model import (
    RecoverFreq,
    HFreq,
    extract_high_low,
    downsample,
    upsample,
    reconstruct_from_high_low,
    CombinedLoss,
)

import pathlib

# =============Config==============
device = torch.device("cuda:0")
IMAGE_ROOT = "../data/imagesTr"
FOURIER_ROOT = "../data/fourier"
LABEL_ROOT = "../data/labelsTr"
BATCH_SIZE = 1
IMAGE_SIZE = 64
TRAIN_STEPS = 10000
SAVE_EVERY = 100
# RESULTS_FOLDER = "./myresults"
INFER_SAVE_DIR = "./myinference_results"
VOLUME_SIZE = 256  # Size of the input volume
# =================================


# =============Random Seed==============
def set_seed(seed=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(10)  # Set a random seed for reproducibility
# =================================


# =============Dataset==============
class NiiDataset(Dataset):
    def __init__(self, image_dir, label_dir, fourier_dir, frequency):
        image_dir = pathlib.Path(image_dir)
        label_dir = pathlib.Path(label_dir)
        fourier_dir = pathlib.Path(fourier_dir)
        self.image_paths = sorted([str(f) for f in list(image_dir.rglob("*.nii.gz"))])
        self.label_paths = sorted([str(f) for f in list(label_dir.rglob("*.nii.gz"))])
        self.fourier_paths = sorted([str(f) for f in list(fourier_dir.rglob("*.pt"))])
        print(f"{len(self.image_paths)} volumes found in {image_dir}")
        self.frequency=frequency

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx])
        label = sitk.ReadImage(self.label_paths[idx])
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        label_array = sitk.GetArrayFromImage(label).astype(np.float32)

        image_x, image_y, image_z = image_array.shape
        if image_x != VOLUME_SIZE or image_y != VOLUME_SIZE or image_z != VOLUME_SIZE:
            image_array = self.resample(
                image, target_shape=(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
            )
        else:

            image_array = torch.tensor(image_array).unsqueeze(0)  # [1, D, H, W]
            label_array = torch.tensor(label_array).unsqueeze(0)  # [1, D, H, W]

        transform = ScaleIntensityRange(
            a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        )
        image_array = transform(image_array)
        # return image_array, label_array  # Return only segmentation as target
        # return image_array  # Return only medical volume
        fourier_tensor = torch.load(
            self.fourier_paths[idx], weights_only=False
        )  # [3, 64, 64, 64] normalized
        assert fourier_tensor.shape == (3, 64, 64, 64), "Fourier tensor shape mismatch"
        if self.frequency == "high":
            fourier_tensor = fourier_tensor[:2]
        elif self.frequency == "low":
            fourier_tensor = fourier_tensor[2:3]
        return (
            image_array,
            fourier_tensor,
        )  # Return both medical volume and Fourier tensor

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


# ================Inference and save functions================
def inference_and_save(diffusion_model, dataset, step, num_samples=2):
    diffusion_model.eval()
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

            fourier_input = torch.cat(
                [hf_process_resample, low_tensor.unsqueeze(0)], dim=0
            )  # [3, 64, 64, 64]

            pred_latent = diffusion_model(fourier_input.unsqueeze(0)).squeeze()
            pred_low = upsample(pred_latent[2], target_size=(256, 256, 256))

            pred_high = HFreq.mag_phase_to_complex(pred_latent[:2])
            pred_high = recover.recover(pred_high)  # [2, 256, 256, 256]
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


# ================Training================
# class Trainer3D:
#     def __init__(self, diffusion, dataset):
#         self.accelerator = Accelerator()
#         self.diffusion = diffusion
#         self.dataloader = DataLoader(
#             dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
#         )
#         self.optimizer = Adam(diffusion.parameters(), lr=2e-4)
#         self.results_folder = pathlib.Path(RESULTS_FOLDER)
#         self.results_folder.mkdir(exist_ok=True)
#         self.step = 0

#         self.diffusion, self.optimizer, self.dataloader = self.accelerator.prepare(
#             self.diffusion, self.optimizer, self.dataloader
#         )

#         self.best_loss = float("inf")
#         self.recover = RecoverFreq()

#     def train(self):
#         device = self.accelerator.device
#         self.diffusion.to(device)

#         for _ in range(TRAIN_STEPS):
#             for img in self.dataloader:
#                 img = img.to(device)
#                 img_denorm = (img + 1.0) / 2.0 * 255.0  # Normalize to [0, 1] range

#                 outputs_high, outputs_low = [], []

#                 # TODO: Add high/low frequency extraction and reconstruction
#                 for b in range(img.shape[0]):
#                     vol = img_denorm[b, 0]  # shape [D, H, W]
#                     high_freq, _ = extract_high_low(vol, high_ratio=0.25)

#                     low_freq = downsample(vol, target_size=(64, 64, 64))
#                     low_tensor = low_freq

#                     hf_process = HFreq.complex_to_mag_phase(high_freq)
#                     high_tensor = hf_process

#                     outputs_high.append(high_tensor)
#                     outputs_low.append(low_tensor)

#                 # x_high = torch.stack(outputs_high)  # [B, 2, D, H, W]
#                 # x_low = torch.stack(outputs_low).unsqueeze(1)

#                 x_high = torch.stack(
#                     [(x - x.min()) / (x.max() - x.min()) * 2 - 1 for x in outputs_high]
#                 )

#                 x_low = torch.stack(
#                     [(x - x.min()) / (x.max() - x.min()) * 2 - 1 for x in outputs_low]
#                 ).unsqueeze(1)

#                 # x_high = (
#                 #     (x_high - x_high.min()) / (x_high.max() - x_high.min())
#                 # ) * 2 - 1
#                 # x_low = ((x_low - x_low.min()) / (x_low.max() - x_low.min())) * 2 - 1

#                 x_input = torch.cat([x_high, x_low], dim=1)  # [B, 3, D, H, W]

#                 with self.accelerator.autocast():
#                     loss = self.diffusion(x_input)
#                 self.accelerator.backward(loss)
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()

#                 if self.accelerator.is_main_process:
#                     if self.step % 10 == 0:
#                         print(f"Step {self.step}: Loss = {loss.item():.4f}")

#                     # Check and save best model
#                     if loss.item() < self.best_loss:
#                         self.best_loss = loss.item()
#                         best_ckpt_path = self.results_folder / f"best_model.pt"
#                         torch.save(self.diffusion.state_dict(), str(best_ckpt_path))
#                         print(
#                             f"New best model saved at step {self.step} with loss {self.best_loss:.4f}"
#                         )
#                     if self.step % SAVE_EVERY == 0:
#                         ckpt_path = self.results_folder / f"model-{self.step}.pt"
#                         torch.save(self.diffusion.state_dict(), str(ckpt_path))
#                         print(f"Saved checkpoint to {ckpt_path}")

#                         inference_and_save(
#                             self.diffusion,
#                             self.dataloader.dataset,
#                             self.step,
#                             num_samples=2,
#                         )

#                 self.step += 1


# ================Model and Diffusion================
# model = SKUNet(
#     spatial_dims=3,
#     in_channels=3,  # noisy volume
#     out_channels=3,  # predicted noise
#     kernel_size=[3, 3, 3, 3, 3, 3],
#     strides=[1, 2, 2, 2, 2, [2, 2, 1]],
#     upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
#     norm_name="instance",
#     deep_supervision=False,
#     res_block=True,
#     time_emb_dim=128,
# )
# diffusion = GaussianDiffusion(model, image_size=IMAGE_SIZE, num_sample_steps=1000)
# ===================================================


# def main():
#     # High Frequency
#     RESULTS_FOLDER = "./twins/myresults_high_freq"
#     model = SKViT3D(dim=64, channels=2, dim_mults=(2, 4, 8, 16))

#     diffusion = GaussianDiffusion3D(
#         model, image_size=IMAGE_SIZE, num_sample_steps=1000, channels=2
#     )

#     dataset = NiiDataset(IMAGE_ROOT, LABEL_ROOT, FOURIER_ROOT, frequency="high")

#     trainer = Trainer3D(
#         diffusion,
#         volume_dataset=dataset,
#         train_batch_size=BATCH_SIZE,
#         train_lr=2e-4,
#         train_num_steps=100000,
#         save_best_and_latest_only=True,
#         save_and_sample_every=SAVE_EVERY,
#         results_folder=RESULTS_FOLDER,
#     )

#     trainer.train()
    
def main():
    # Low Frequency
    RESULTS_FOLDER = "./twins/myresults_low_freq"
    model = SKViT3D(dim=64, channels=1, dim_mults=(2, 4, 8, 16))

    diffusion = GaussianDiffusion3D(
        model, image_size=IMAGE_SIZE, num_sample_steps=1000, channels=1
    )

    dataset = NiiDataset(IMAGE_ROOT, LABEL_ROOT, FOURIER_ROOT, frequency="low")

    trainer = Trainer3D(
        diffusion,
        volume_dataset=dataset,
        train_batch_size=BATCH_SIZE,
        train_lr=2e-4,
        train_num_steps=100000,
        save_best_and_latest_only=True,
        save_and_sample_every=SAVE_EVERY,
        results_folder=RESULTS_FOLDER,
    )

    trainer.train()


if __name__ == "__main__":
    # # Create dataset
    # dataset = NiiDataset(IMAGE_ROOT, LABEL_ROOT)

    # # Create trainer
    # trainer = Trainer3D(diffusion, dataset)

    # # Start training
    # trainer.train()

    main()
