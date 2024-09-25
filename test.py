import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

class GaussianDiffusion:
    def __init__(self, denoise_fn, image_size):
        self.denoise_fn = denoise_fn
        self.image_size = image_size
        # Initialize betas, alphas, and alphas_cumprod as needed
        # For example:
        self.betas = np.linspace(1e-4, 2e-2, 2000)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def space_timesteps(self, num_timesteps=2000, section_counts='ddim250'):
        if isinstance(section_counts, str):
            if section_counts.startswith("ddim"):
                desired_count = int(section_counts[len("ddim"):])
                for i in range(1, num_timesteps):
                    if len(range(0, num_timesteps, i)) == desired_count:
                        return set(range(0, num_timesteps, i))
                raise ValueError(
                    f"Cannot create exactly {desired_count} steps with an integer stride"
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
                    f"Cannot divide section of {size} steps into {section_count}"
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
        use_timesteps = self.space_timesteps(2000, "ddim250")
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(self.betas)

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                # 来自 beta 与 alpha 之间的关系式
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        # 确保 new_betas 和 timestep_map 的长度与 use_timesteps 一致
        assert len(new_betas) == len(self.use_timesteps), (
            f"Expected new_betas length to be {len(self.use_timesteps)}, but got {len(new_betas)}"
        )
        assert len(self.timestep_map) == len(new_betas), (
            f"Expected timestep_map length to be {len(new_betas)}, but got {len(self.timestep_map)}"
        )

        tensor_list_cpu = [torch.tensor(tensor).cpu() for tensor in new_betas]
        self.betas = np.array(new_betas)
        return tensor_list_cpu

# 测试代码
if __name__ == "__main__":
    base_diffusion = GaussianDiffusion(None, 32)
    tensor_list = base_diffusion.scale_betas()
    print(f"Length of new_betas: {len(base_diffusion.betas)}")
    print(f"Length of timestep_map: {len(base_diffusion.timestep_map)}")
    print(f"Length of use_timesteps: {len(base_diffusion.use_timesteps)}")
