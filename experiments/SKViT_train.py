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
from denoising_diffusion_pytorch.sketch_diffusion import SKViT3D, GaussianDiffusion3D
from denoising_diffusion_pytorch import Trainer3D

# from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D
from torch.optim import Adam
from accelerate import Accelerator

import pathlib

# =============Config==============
device = torch.device("cuda:0")
IMAGE_ROOT = "../data/imagesTr"
LABEL_ROOT = "../data/labelsTr"
BATCH_SIZE = 4
IMAGE_SIZE = 128
TRAIN_STEPS = 10000
SAVE_EVERY = 500
RESULTS_FOLDER = "./results_3d_100000_epochs"
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
            a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
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


def main():
    model = SKViT3D(dim=16, channels=1)

    diffusion = GaussianDiffusion3D(model, image_size=128, num_sample_steps=1000)

    dataset = NiiDataset(IMAGE_ROOT, LABEL_ROOT)

    trainer = Trainer3D(
        diffusion,
        volume_dataset=dataset,
        train_batch_size=BATCH_SIZE,
        train_lr=1e-4,
        train_num_steps=100000,
        save_best_and_latest_only=True,
        save_and_sample_every=SAVE_EVERY,
        results_folder=RESULTS_FOLDER,
    )

    trainer.train()

    # ================test the function===============
    # x = torch.randn(3, 1, 128, 128, 128).float().to(device)  # Example input
    # model = model.to(device)
    # t = torch.randint(0, 1000, (3,)).float().to(device)  # Example timesteps
    # output = model(x, t)  # Forward pass
    # print(output.shape)  # Should be [5, 1, 128, 128, 128]


# CUDA_VISIBLE_DEVICES=2 python /ehome/mingqi/aorta_diffusion/denoising-diffusion-pytorch/diffusion_small.py

if __name__ == "__main__":
    main()
