"""
Gürültü Zaman Çizelgesi (Noise Schedule) Modülü (Day 172 - FAZ 9).
Doğrusal (Linear) ve Kosinüs (Cosine) çizelgeleri ile beta, alpha, alpha_bar hesaplaması ve ileri difüzyon.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


class GurultuZamanCizelgesi:
    """Linear ve Cosine Difüzyon Zaman Çizelgesi Yöneticisi."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._kosinus_betalari_uret(num_timesteps)
        else:
            raise ValueError(f"Bilinmeyen çizelge tipi: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # İleri difüzyon için karekökler
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _kosinus_betalari_uret(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Nichol & Dhariwal Cosine Noise Schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def ileri_difuzyon(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        gurultu: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q(z_t | z_0) = N(z_t; sqrt(alpha_bar_t) * z_0, (1 - alpha_bar_t) * I)
        z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if gurultu is None:
            gurultu = torch.randn_like(z_0)

        # t tensorunu yayınla (broadcast)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(z_0.device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(z_0.device)

        z_t = sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * gurultu
        return z_t, gurultu
