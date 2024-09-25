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


# gaussian diffusion trainer class

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

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise['noise']
            
    # def predict_start_from_noise(self, x_t, t, noise):
    #     return self.sqrt_recip_alphas_cumprod[t] * x_t - \
    #         self.sqrt_recipm1_alphas_cumprod[t] * noise

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
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level,step_t=t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

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
        if current_step < 100000:
        # if current_step < 100:
            loss = loss1 + loss3
            print("loss的组成为 loss1 + loss3")
        else:
            loss = loss1 + loss2 + loss3 + loss4
            print("loss的组成为 loss1 + loss2 + loss3 + loss4")
        return loss

    def forward(self, x, current_step, *args, **kwargs):
        return self.p_losses(x,current_step, *args, **kwargs)








#     #for test
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

#     HR_file = '/home/fiko/Code/DATASET/Alsat-2B-main/sr3_Alsat-2B_test_32_128/hr_128/0HR_0.png'
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
#     # pretrained_dict = torch.load(pretrained_model_path)

#     model = UNet()

#     # 获取模型的当前状态字典
#     model_dict = model.state_dict()

#     # 将预训练参数加载到模型中
#     # 注意：确保预训练模型的键和你的模型的键匹配
#     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # model_dict.update(pretrained_dict)
#     # model.load_state_dict(model_dict)

#     # 设置模型为评估模式
#     model.eval()
#     noise  = model(torch.cat([x_SR, x_noisy], dim=1),sqrt_alpha_cumprod)
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

