"""
DDIM Hızlı Örnekleme Zamanlayıcısı Modülü (Day 173 - FAZ 9).
Denoising Diffusion Implicit Models (Song et al., 2020) ile 1000 adımı 20-50 deterministik adıma indirger.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class DDIMZamanlayici:
    """Deterministik DDIM (eta=0.0) Hızlı Örnekleme Zamanlayıcısı."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 20,
        eta: float = 0.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta  # eta=0.0 saf deterministik ODE, eta=1.0 klasik DDPM

        # Eşit aralıklı alt zaman adımları
        step_ratio = num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_train_timesteps, step_ratio).flip(0)

        # 1000 adımlı temel alpha_bar serisi
        betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def ornekleme_adimi(
        self,
        z_t: torch.Tensor,
        eps_guided: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        DDIM Adım Formülü (Song et al., 2020):
        1. Tahmin edilen temiz z_0:
           pred_z_0 = (z_t - sqrt(1 - alpha_bar_t) * eps_guided) / sqrt(alpha_bar_t)
        2. Yön tensörü (Direction pointing to z_t):
           dir_z_t = sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_guided
        3. Bir sonraki z_{t-1}:
           z_{t-1} = sqrt(alpha_bar_{t-1}) * pred_z_0 + dir_z_t + sigma_t * epsilon
        """
        curr_t = self.timesteps[t_idx].item()
        prev_t = self.timesteps[t_idx + 1].item() if t_idx + 1 < len(self.timesteps) else 0

        alpha_bar_t = self.alphas_cumprod[curr_t]
        alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else torch.tensor(1.0)

        # sigma_t (eta=0.0 iken sigma_t=0)
        sigma_t = self.eta * torch.sqrt(
            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
        )

        # 1. Kestirilen Temiz z_0
        pred_z_0 = (z_t - torch.sqrt(1.0 - alpha_bar_t) * eps_guided) / torch.sqrt(alpha_bar_t)

        # 2. Deterministik Yön Vektörü
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * eps_guided

        # 3. Geri Difüzyon Güncellemesi
        z_prev = torch.sqrt(alpha_bar_prev) * pred_z_0 + dir_xt
        if self.eta > 0.0:
            z_prev = z_prev + sigma_t * torch.randn_like(z_t)

        return z_prev
