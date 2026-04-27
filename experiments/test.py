import torch
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append("..")
sys.path.insert(1, "./")

from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion
from denoising_diffusion_pytorch.model_utils import SKUNet

# =============Config==============
IMAGE_SIZE = 128
SAVE_STEPS = 100
NUM_SAMPLE_STEPS = 1000
CHECKPOINT_PATH = "./results/model-7000.pt"
OUTPUT_DIR = "./synthesized_volumes"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================


def inverse_transform(tensor):
    """
    Convert from [-1, 1] back to [-1000, 1000] CT intensity range
    """
    return (tensor + 1.0) * 1000.0


def save_nifti(volume_tensor, filename, spacing=(1.0, 1.0, 1.0)):
    """
    Save a 3D tensor as a NIfTI file with spacing info.
    """
    volume = volume_tensor.squeeze().cpu().numpy()
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, filename)


def main():
    # Initialize model
    model = SKUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
        time_emb_dim=128,
    )
    diffusion = GaussianDiffusion(
        model, image_size=IMAGE_SIZE, num_sample_steps=NUM_SAMPLE_STEPS
    )
    diffusion.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    diffusion.to(DEVICE)
    model.eval()
    diffusion.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Start sampling
    with torch.no_grad():
        noisy = torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
        xt = noisy.clone()

        for t in tqdm(reversed(range(NUM_SAMPLE_STEPS)), desc="Sampling"):
            t_cur = torch.tensor(t, device=DEVICE)
            t_next = torch.tensor(max(t - 1, 0), device=DEVICE)
            xt = diffusion.p_sample(xt, t_cur, t_next)
            
            print(f"xt min: {xt.min().item():.2f}, max: {xt.max().item():.2f}")

            if t % SAVE_STEPS == 0 or t == NUM_SAMPLE_STEPS - 1:
                restored = inverse_transform(xt.clone())
                save_nifti(restored, Path(OUTPUT_DIR) / f"step_{t:04d}.nii.gz")

        # Save final result
        restored = inverse_transform(xt)
        save_nifti(restored, os.path.join(OUTPUT_DIR, "final_generated.nii.gz"))
        print("Saved final result.")


if __name__ == "__main__":
    main()
