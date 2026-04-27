import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.special import expm1
from torch.amp import autocast
from functools import wraps

# ========== Helper Functions ==========

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def cast_tuple(t, l=1):
    return (t,) * l if not isinstance(t, tuple) else t

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# ========== RMSNorm ==========

class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, normalize_dim=2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1
        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        scale = self.g.view(1, -1, *((1,) * (x.ndim - 2))) if self.scale else 1
        return F.normalize(x, dim=self.normalize_dim) * scale * (x.shape[self.normalize_dim] ** 0.5)

# ========== 3D Building Blocks ==========

class Block3D(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out, normalize_dim=1)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block3D(dim, dim_out)
        self.block2 = Block3D(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Downsample3D(nn.Module):
    def __init__(self, dim, dim_out, factor=2):
        super().__init__()
        self.op = nn.Conv3d(dim, dim_out, kernel_size=factor, stride=factor)

    def forward(self, x):
        return self.op(x)

class Upsample3D(nn.Module):
    def __init__(self, dim, dim_out, factor=2):
        super().__init__()
        self.op = nn.ConvTranspose3d(dim, dim_out, kernel_size=factor, stride=factor)

    def forward(self, x):
        return self.op(x)

# ========== SKViT 3D ==========

class SKViT3D(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1, time_dim_mult=4):
        super().__init__()
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(channels, init_dim, kernel_size=3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * time_dim_mult
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(nn.ModuleList([
                ResnetBlock3D(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock3D(dim_in, dim_in, time_emb_dim=time_dim),
                Downsample3D(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock3D(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock3D(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out):
            self.ups.append(nn.ModuleList([
                Upsample3D(dim_out, dim_in),
                ResnetBlock3D(dim_in * 2, dim_in, time_emb_dim=time_dim),
                ResnetBlock3D(dim_in * 2, dim_in, time_emb_dim=time_dim)
            ]))

        self.final_block = ResnetBlock3D(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, default(out_dim, channels), 1)

    def forward(self, x, time):
        t = self.time_mlp(time.view(-1, 1))

        x = self.init_conv(x)
        r = x.clone()
        h = []

        for block1, block2, down in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for up, block1, block2 in self.ups:
            x = up(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)

        x = torch.cat((x, r), dim=1)
        x = self.final_block(x, t)
        return self.final_conv(x)

# ========== Log SNR Schedules ==========

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

# ========== Gaussian Diffusion 3D ==========

class GaussianDiffusion3D(nn.Module):
    def __init__(
        self,
        model: SKViT3D,
        *,
        image_size,
        channels=1,
        pred_objective="v",
        noise_schedule=logsnr_schedule_cosine,
        num_sample_steps=500,
        clip_sample_denoised=True,
        min_snr_loss_weight=True,
        min_snr_gamma=5,
    ):
        super().__init__()
        assert pred_objective in {"v", "eps"}
        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.pred_objective = pred_objective
        self.log_snr = noise_schedule
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(torch.sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
        pred = self.model(x, batch_log_snr)

        if self.pred_objective == "v":
            x_start = alpha * x - sigma * pred
        else:
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1.0, 1.0)
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        model_mean, model_variance = self.p_mean_variance(x, time, time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)
        for i in range(self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)
        img.clamp_(-1.0, 1.0)
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def sample(self, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size, self.image_size))

    @autocast(device_type='cuda', enabled=False)
    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = torch.sqrt(log_snr_padded.sigmoid()), torch.sqrt((-log_snr_padded).sigmoid())
        return x_start * alpha + noise * sigma, log_snr

    def p_losses(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr)

        if self.pred_objective == "v":
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = torch.sqrt(padded_log_snr.sigmoid()), torch.sqrt((-padded_log_snr).sigmoid())
            target = alpha * noise - sigma * x_start
        else:
            target = noise

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = loss.mean(dim=[1, 2, 3, 4])
        snr = log_snr.exp()

        if self.min_snr_loss_weight:
            snr = snr.clamp(max=self.min_snr_gamma)

        if self.pred_objective == "v":
            loss_weight = snr / (snr + 1)
        else:
            loss_weight = snr.reciprocal()

        return (loss * loss_weight).mean()

    def forward(self, img):
        b, c, d, h, w = img.shape
        assert d == h == w == self.image_size
        img = normalize_to_neg_one_to_one(img)
        times = torch.rand((b,), device=self.device)
        return self.p_losses(img, times)