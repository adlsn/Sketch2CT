import torch
import numpy as np
import sys
sys.path.append("..")
sys.path.insert(1,'./')
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from registration.mesh_operations.normalizer import *
import pathlib

device = torch.device("cuda:0")

data_dir = './data'
data = np.load('/ehome/mingqi/aorta_diffusion/data/original_data/originalcl.npy')  # shape: (22, 16, 3)

data = torch.tensor(data, dtype=torch.float32)
data_norm = (data - data.min()) / (data.max() - data.min())
data_norm = data_norm.transpose(1,2) #shape: (22,3,16)
list = []
for ii in range(0,100):
    list.append(data_norm)
data_norm = torch.stack(list).view(2200,3,16)

model = Unet1D(
    dim = 64,           
    dim_mults = (1, 2, 4, 8),  
    channels = 3
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = data_norm.shape[-1],  
    timesteps = 1000,         
    objective = 'pred_v'       
)

dataset = Dataset1D(data_norm)

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 22,
    train_lr = 8e-5,
    train_num_steps = 700000,   
    gradient_accumulate_every = 2, 
    ema_decay = 0.995,
    amp = True
)

trainer.train()



# CUDA_VISIBLE_DEVICES=2 python /ehome/mingqi/aorta_diffusion/denoising-diffusion-pytorch/diffusion_small.py