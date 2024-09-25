import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,    
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
    
    def space_timesteps(self,num_timesteps, section_counts):
        if isinstance(section_counts, str):
            if section_counts.startswith("ddim"):
                desired_count = int(section_counts[len("ddim") :]) # 取出ddim后面的数字 250
                for i in range(1, num_timesteps):
                    if len(range(0, num_timesteps, i)) == desired_count:
                        return set(range(0, num_timesteps, i))
                raise ValueError(
                    f"cannot create exactly {num_timesteps} steps with an integer stride"
                )
            section_counts = [int(x) for x in section_counts.split(",")]
        size_per = num_timesteps // len(section_counts)
        extra = num_timesteps % len(section_counts)
        start_idx = 0
        all_steps = []
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            all_steps += taken_steps
            start_idx += size
        return set(all_steps)
        
    def scale_betas(self):
        use_timesteps= self.space_timesteps(2000, "ddim250")
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(self.betas)

        base_diffusion = GaussianDiffusion(None,32)  # pylint: disable=missing-kwoa
        #  计算全新采样时刻后的betas
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                # 来自beta与alpha之间的关系式
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i) #[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, ...]
                
        tensor_list_cpu = [tensor.cpu() for tensor in new_betas]
        self.betas = np.array(new_betas) # len(new_betas) 等于 250
        super().__init__(self.use_timesteps,self.timestep_map, self.original_num_steps,self.betas)
        return tensor_list_cpu
    
    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise['noise']
            
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level, step_t=t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def ddim_sample(self, x, t, ddim_steps, clip_denoised=True, condition_x=None):
        batch_size = x.shape[0]
        eta = 0  # Noise factor
        ddim_timestep_seq = torch.linspace(0, self.num_timesteps - 1, ddim_steps).long()
        ddim_timestep_next_seq = torch.cat([ddim_timestep_seq[1:], torch.tensor([0])])

        img = x
        for current_t, next_t in zip(reversed(ddim_timestep_seq), reversed(ddim_timestep_next_seq)):
            model_mean, model_log_variance = self.p_mean_variance(
                x=img, t=current_t, clip_denoised=clip_denoised, condition_x=condition_x)
            alpha_cumprod_t = self.alphas_cumprod[current_t]
            alpha_cumprod_next = self.alphas_cumprod[next_t]

            # Predict x_0
            pred_x0 = model_mean

            # Compute the mean
            mean = (alpha_cumprod_next.sqrt() * pred_x0 +
                    (1 - alpha_cumprod_next).sqrt() * torch.randn_like(img))

            noise = torch.randn_like(img) if eta > 0 and next_t != 0 else torch.zeros_like(img)
            sigma = eta * torch.sqrt((1 - alpha_cumprod_next) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_next)
            img = mean + sigma * noise

        return img

    @torch.no_grad()
    def ddim_sample_loop(self, x_in, ddim_steps, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, ddim_steps)), desc='sampling loop time step', total=ddim_steps):
                img = self.ddim_sample(img, i, ddim_steps)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, ddim_steps)), desc='sampling loop time step', total=ddim_steps):
                img = self.ddim_sample(img, i, ddim_steps, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False, ddim_steps=None):
        image_size = self.image_size
        channels = self.channels
        if ddim_steps is None:
            return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)
        else:
            return self.ddim_sample_loop((batch_size, channels, image_size, image_size), ddim_steps, continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, ddim_steps=None):
        if ddim_steps is None:
            return self.p_sample_loop(x_in, continous)
        else:
            return self.ddim_sample_loop(x_in, ddim_steps, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, current_step,noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn( # 在模型的forward中传入参数
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod, step_t = t)
            
        
        # 根据iter更改loss组成
        loss1 = self.loss_func(noise, x_recon['noise'])
        loss2 = self.loss_func(x_start, x_recon['X_0'])
        loss3 = self.loss_func(noise.var(), x_recon['noise'].var())
        loss4 = self.loss_func(x_start.var(), x_recon['X_0'].var())
        if current_step < 140000:
        # if current_step < 100:
            loss = loss1 + loss3
            # print("loss的组成为 loss1 + loss3")
        else:
            loss = loss1 + loss2 + loss3 + loss4
            # print("loss的组成为 loss1 + loss2 + loss3 + loss4")
        return loss

    def forward(self, x, current_step, *args, **kwargs):
        return self.p_losses(x,current_step, *args, **kwargs)






import math
import torch
from functools import partial
import numpy as np
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from torch import device, nn, einsum

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel) # [32,512]
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False) # [512,1536]
        self.out = nn.Conv2d(in_channel, in_channel, 1) # [512,512]

    def forward(self, input):
        batch, channel, height, width = input.shape # [8,512,16,16]
        n_head = self.n_head
        head_dim = channel // n_head # 512

        norm = self.norm(input) # [8,512,16,16]
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width) # [8,1,1536,16,16]
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx  [8,1,512,16,16]   [8,1,512,16,16]   [8,1,512,16,16]

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1) # [8,1,16,16,256]
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width) # [8,1,16,16,16,16] [8,1,8,8,8,8]

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width)) # [8,512,16,16]

        return out + input  # [8,512,16,16] [8,512,8,8]


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


def predict_start_from_noise(self, x_t, t, noise):
    return self.sqrt_recip_alphas_cumprod[t] * x_t - \
        self.sqrt_recipm1_alphas_cumprod[t] * noise


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=[16],
        res_blocks=2,
        dropout=0,
        with_noise_level_emb=True, # time_embedding
        image_size=128,
        schedule_opt=None
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)  # [1,2,4,8,8] length等于5
        pre_channel = inner_channel   # 64
        feat_channels = [pre_channel] # [64]
        now_res = image_size # 128
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res) # 判断是否为attn_res 此处是否为16
            channel_mult = inner_channel * channel_mults[ind] # 64 * 1 2 4 8 8
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        
        if schedule_opt is not None:
            pass
            self.set_new_noise_schedule(schedule_opt,'cuda')
        
        self.set_new_noise_schedule(schedule_opt,'cuda')
        
    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule='linear',
            n_timestep=250,
            linear_start=1e-4,
            linear_end=2e-2)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                            to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                            to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        

    def forward(self, x, time, step_t):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        
        LR, HR_noisey = torch.chunk(x, 2, dim=1)
        X_0 = torch.zeros_like(HR_noisey)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
                
        noise = self.final_conv(x)
        if step_t == 2000:
            step_t = step_t -1
        for i in range(x.shape[0]):
            X_0[i] = predict_start_from_noise(self,x_t = HR_noisey[i], t = step_t, noise= noise[i])
        
        
        return {'noise': noise,'X_0': X_0}
        # return self.final_conv(x)



if __name__ == '__main__':
    # Using GaussianDiffusion with DDIM acceleration
    dm = GaussianDiffusion(None,32)
    dm.set_new_noise_schedule(schedule_opt={
            'schedule': 'linear',
            'n_timestep': 1000,
            'linear_start': 1e-4,
            'linear_end': 2e-2,
        },device='cpu')
    new_betas  = dm.scale_betas()
    print(new_betas)


        
#     def test_q_sample(self, x_start, t, noise, device='cpu'):
#         to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
#         betas = make_beta_schedule(
#             schedule='linear',
#             n_timestep=2000)
#         betas = betas.detach().cpu().numpy() if isinstance(
#             betas, torch.Tensor) else betas
#         alphas = 1. - betas
#         alphas_cumprod = np.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
#         self.sqrt_alphas_cumprod_prev = np.sqrt(
#             np.append(1., alphas_cumprod))

#         timesteps, = betas.shape
#         self.num_timesteps = int(timesteps)
#         self.register_buffer('betas', to_torch(betas))
#         self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
#         self.register_buffer('alphas_cumprod_prev',
#                              to_torch(alphas_cumprod_prev))

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer('sqrt_alphas_cumprod',
#                              to_torch(np.sqrt(alphas_cumprod)))
#         self.register_buffer('sqrt_one_minus_alphas_cumprod',
#                              to_torch(np.sqrt(1. - alphas_cumprod)))
#         self.register_buffer('log_one_minus_alphas_cumprod',
#                              to_torch(np.log(1. - alphas_cumprod)))
#         self.register_buffer('sqrt_recip_alphas_cumprod',
#                              to_torch(np.sqrt(1. / alphas_cumprod)))
#         self.register_buffer('sqrt_recipm1_alphas_cumprod',
#                              to_torch(np.sqrt(1. / alphas_cumprod - 1)))

#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         posterior_variance = betas * \
#             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#         # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
#         self.register_buffer('posterior_variance',
#                              to_torch(posterior_variance))
#         # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
#         self.register_buffer('posterior_log_variance_clipped', to_torch(
#             np.log(np.maximum(posterior_variance, 1e-20))))
#         self.register_buffer('posterior_mean_coef1', to_torch(
#             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
#         self.register_buffer('posterior_mean_coef2', to_torch(
#             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))    

#         [b, c, h, w] = x_start.shape
#         #t = np.random.randint(1, self.num_timesteps + 1)
#         continuous_sqrt_alpha_cumprod = torch.FloatTensor(
#             np.random.uniform(
#                 self.sqrt_alphas_cumprod_prev[t-1],
#                 self.sqrt_alphas_cumprod_prev[t],
#                 size=b
#             )
#         ).to('cpu')

#         continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
#             b, -1)
        
#         x_noisy = self.q_sample(
#             x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
#         return x_noisy,continuous_sqrt_alpha_cumprod

#     def forward(self, x, *args, **kwargs):
#         return self.p_losses(x, *args, **kwargs)


# def tensor2image(t):
#     t = t.detach().numpy()
#     t = t.squeeze()
#     t = (t+1)/2
#     t = np.clip(t, 0, 1)
#     t = t*255
#     t = t.astype(np.uint8)
#     t = t.transpose(1,2,0)
#     t = t[:,:,(2,1,0)]
#     return t

# if __name__=='__main__':

#     '''
#     import matplotlib.pyplot as plt

#     #测试噪声变化方案
#     #betas = make_beta_schedule('linear', 50, linear_start=1e-4, linear_end=0.05 )
#     betas = make_beta_schedule('linear', 1000, linear_start=1e-4, linear_end=0.005 )
#     #print(betas)
#     alphas = 1. - betas
#     alphas_cumprod = np.cumprod(alphas, axis=0)
#     #print( alphas_cumprod )
#     #print( np.append(1., alphas_cumprod) )

#     plt.plot(alphas_cumprod,'r.')

#     #alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
#     sqrt_alphas_cumprod_prev = np.sqrt(
#             np.append(1., alphas_cumprod) )
#     #print(sqrt_alphas_cumprod_prev)

    
#     #plt.plot(alphas_cumprod,'r-')
#     plt.plot(sqrt_alphas_cumprod_prev,'b-')
#     plt.show()
#     '''

#     from PIL import Image
#     import torchvision
#     import cv2
#     import numpy as np
#     from unet import UNet

#     totensor = torchvision.transforms.ToTensor()

#     HR_file = '/home/fiko/Code/Super_Resolution/End2End_SR/dataset/celebahq_16_128/hr_128/00031.png'
#     SR_file = '/home/fiko/Code/Super_Resolution/End2End_SR/dataset/celebahq_16_128/sr_16_128/00031.png' 

#     x_HR = Image.open(HR_file).convert("RGB")
#     x_SR = Image.open(SR_file).convert("RGB")
    
#     x_HR = totensor(x_HR)
#     x_SR = totensor(x_SR)
#     print(x_SR.max(), x_SR.min())
#     x_HR = x_HR*2-1
#     x_SR = x_SR*2-1
    
#     x_HR = x_HR.unsqueeze(0)
#     x_SR = x_SR.unsqueeze(0)
#     print(x_HR.shape)

#     y_noise = torch.randn_like(x_HR)
    
#     dm = GaussianDiffusion(None, 32)

#     #前向加噪声
#     #dm.set_new_noise_schedule('linear', 'cpu')
#     betas = make_beta_schedule(
#         schedule='linear',
#         n_timestep=2000)
#     betas = betas.detach().cpu().numpy() if isinstance(
#         betas, torch.Tensor) else betas
#     alphas = 1. - betas
#     alphas_cumprod = np.cumprod(alphas, axis=0)
#     use_timesteps = dm.space_timesteps(num_timesteps = 2000, section_counts="ddim250")
#     new_betas = dm.scalepace(use_timesteps, alphas_cumprod)
#     t = 100
#     x_noisy,sqrt_alpha_cumprod = dm.test_q_sample(x_HR, t, y_noise)
#     x_t = x_noisy

#     #采样
#     '''
#     在这里load模型，返回noise
#     测试验证
#     if 通过： U-net中forward返回均值和方差 做new loss
#     '''
    
#     # pretrained_model_path = '/home/fiko/Code/Super_Resolution/End2End_SR/experiments/loss1__loss1+loss2/resume/I190000_E2090_gen.pth'
#     pretrained_model_path = '/home/fiko/Code/Super_Resolution/End2End_SR/experiments/sr_Alsat_240507_205054/checkpoint/I100_E1_gen.pth'
#     pretrained_dict = torch.load(pretrained_model_path)

#     model = UNet()

#     # 获取模型的当前状态字典
#     model_dict = model.state_dict()

#     # 将预训练参数加载到模型中
#     # 注意：确保预训练模型的键和你的模型的键匹配
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)

#     # 设置模型为评估模式
#     model.eval()
#     noise  = model(torch.cat([x_SR, x_noisy], dim=1),sqrt_alpha_cumprod, t)
#     print('模型预测的噪声均值为',noise.mean(),'模型预测的噪声方差为',noise.var())
#     print('原始的噪声均值为',y_noise.mean(),'原始的噪声方差为',y_noise.var())
#     x_0 = dm.predict_start_from_noise(x_t, t, noise)
    
#     #可视化
#     x_t = tensor2image(x_t)
#     x_0 = tensor2image(x_0)
#     #print(x_noisy.shape)
#     cv2.imwrite('x_t.jpg', x_t)
#     cv2.imwrite('x_0.jpg', x_0)
#     #out = Image.fromarray(x_noisy)
#     #out.save('x_t.jpg')



