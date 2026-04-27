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

sys.path.append("..")
sys.path.insert(1, "./")
from torch.utils.data import Dataset, DataLoader
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion
from denoising_diffusion_pytorch.model_utils import DynUNetWithTimeEmbedding, SKUNet

# from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D
from torch.optim import Adam
from accelerate import Accelerator

import pathlib

# =============Config==============
device = torch.device("cuda:0")
IMAGE_ROOT = "../data/imagesTr"
LABEL_ROOT = "../data/labelsTr"
BATCH_SIZE = 6
IMAGE_SIZE = 128
TRAIN_STEPS = 10000
SAVE_EVERY = 500
RESULTS_FOLDER = "./results"
# =================================


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

            image_array = torch.tensor(image_array).unsqueeze(0)  # [1, D, H, W]
            label_array = torch.tensor(label_array).unsqueeze(0)  # [1, D, H, W]

        transform = ScaleIntensityRange(
            a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        )
        image_array = transform(image_array)
        # return image_array, label_array  # Return only segmentation as target
        return image_array  # Return only medical volume

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


# ================Training================
class Trainer3D:
    def __init__(self, diffusion, dataset):
        self.accelerator = Accelerator()
        self.diffusion = diffusion
        self.dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
        )
        self.optimizer = Adam(diffusion.parameters(), lr=2e-4)
        self.results_folder = pathlib.Path(RESULTS_FOLDER)
        self.results_folder.mkdir(exist_ok=True)
        self.step = 0

        self.diffusion, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.diffusion, self.optimizer, self.dataloader
        )
        
        self.best_loss = float("inf")

    def train(self):
        device = self.accelerator.device
        self.diffusion.to(device)

        for _ in range(TRAIN_STEPS):
            for img in self.dataloader:
                img = img.to(device)
                with self.accelerator.autocast():
                    loss = self.diffusion(img)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                

                if self.accelerator.is_main_process:
                    if self.step % 10 == 0:
                        print(f"Step {self.step}: Loss = {loss.item():.4f}")
                        
                    # Check and save best model
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        best_ckpt_path = self.results_folder / f"best_model.pt"
                        torch.save(self.diffusion.state_dict(), str(best_ckpt_path))
                        print(f"New best model saved at step {self.step} with loss {self.best_loss:.4f}")
                    if self.step % SAVE_EVERY == 0:
                        ckpt_path = self.results_folder / f"model-{self.step}.pt"
                        torch.save(self.diffusion.state_dict(), str(ckpt_path))
                        print(f"Saved checkpoint to {ckpt_path}")
                self.step += 1


# ================Model and Diffusion================
model = SKUNet(
    spatial_dims=3,
    in_channels=1,  # noisy volume
    out_channels=1,  # predicted noise
    kernel_size=[3, 3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2, [2, 2, 1]],
    upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
    norm_name="instance",
    deep_supervision=False,
    res_block=True,
    time_emb_dim=128,
)
diffusion = GaussianDiffusion(model, image_size=IMAGE_SIZE, num_sample_steps=1000)
# ===================================================

if __name__ == "__main__":
    # Create dataset
    dataset = NiiDataset(IMAGE_ROOT, LABEL_ROOT)

    # Create trainer
    trainer = Trainer3D(diffusion, dataset)

    # Start training
    trainer.train()
