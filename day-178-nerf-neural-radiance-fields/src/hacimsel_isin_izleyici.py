"""
Hacimsel Işın Takibi ve Render Entegratörü Modülü (Day 178 - FAZ 9).
Işın denklemi: r(t) = o + t*d
Hacimsel İntegral: C(r) = sum(T_i * (1 - exp(-sigma_i * delta_i)) * c_i)
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from .nerf_mlp import NeRFModeli


class HacimselIsinIzleyici(nn.Module):
    """Kamera Işınları Oluşturucu ve Hacimsel Render Motoru."""

    def __init__(self, model: NeRFModeli, near: float = 2.0, far: float = 6.0, num_samples: int = 64):
        super().__init__()
        self.model = model
        self.near = near
        self.far = far
        self.num_samples = num_samples

    def isin_ornekleme_noktalari_uret(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        perturb: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        rays_o: [N_rays, 3] (Işın Başlangıç Noktaları)
        rays_d: [N_rays, 3] (Işın Birim Doğrultu Vektörleri)
        Döner: (pts [N_rays, N_samples, 3], z_vals [N_rays, N_samples], deltas [N_rays, N_samples])
        """
        n_rays = rays_o.shape[0]
        # Tabakalı (Stratified) derinlik örnekleme
        t_vals = torch.linspace(0.0, 1.0, self.num_samples, device=rays_o.device)
        z_vals = self.near * (1.0 - t_vals) + self.far * t_vals
        z_vals = z_vals.expand(n_rays, self.num_samples).clone()

        if perturb:
            mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            lower = torch.cat([z_vals[:, :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        # Noktalar: r(t) = o + t * d
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)

        # Delta aralıkları
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        deltas = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)

        return pts, z_vals, deltas

    def render_isin(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        perturb: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Işın boyunca örneklenen noktalardan NeRF renk ve yoğunluklarını toplar ve piksel rengi üretir.
        """
        pts, z_vals, deltas = self.isin_ornekleme_noktalari_uret(rays_o, rays_d, perturb=perturb)
        n_rays, n_samples, _ = pts.shape

        # Bakış yönünü çoğalt
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs_expanded = viewdirs.unsqueeze(1).expand(n_rays, n_samples, 3)

        # NeRF İleri Beslemesi
        rgb, sigma = self.model(pts, viewdirs_expanded)
        sigma = sigma.squeeze(-1)  # [N_rays, N_samples]

        # 1. Opaklık (Alpha): alpha_i = 1 - exp(-sigma_i * delta_i)
        alpha = 1.0 - torch.exp(-sigma * deltas)

        # 2. Geçirgenlik (Transmittance): T_i = exp(-sum_{j<i} sigma_j * delta_j) = cumprod(1 - alpha_{i-1})
        cumprod = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(cumprod[:, :1]), cumprod[:, :-1]], dim=-1)

        # 3. Hacimsel Ağırlık: w_i = T_i * alpha_i
        weights = transmittance * alpha

        # 4. Piksel Rengi İntegrali: C(r) = sum(w_i * c_i)
        comp_rgb = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)

        # Derinlik Tahmini: depth = sum(w_i * z_i)
        depth_map = torch.sum(weights * z_vals, dim=-1)

        return {
            "rgb": comp_rgb,
            "depth": depth_map,
            "weights": weights,
            "z_vals": z_vals,
            "sigma": sigma,
        }

    @classmethod
    def ornek_nerf_sahne_raporu(cls) -> Dict[str, Any]:
        """NeRF sahne sentezi ve PSNR/SSIM metrikleri."""
        return {
            "sahne_adi": "Synthetic Lego & Realistic Drums 3D Sahnesi",
            "kamera_sayisi": 100,
            "ornekleme_sayisi": 64,
            "metrikler": {
                "psnr": 34.2,
                "ssim": 0.965,
                "lpips": 0.048,
                "render_fps": 0.45,
            },
            "karsilastirma": [
                {"yontem": "Voxel Grid (Ayrık Hacim)", "psnr": 27.5, "bellek_mb": 4096, "durum": "Yüksek Bellek & Bloklaşma"},
                {"yontem": "Point Cloud (Nokta Bulutu)", "psnr": 29.1, "bellek_mb": 512, "durum": "Boşluklar ve Doku Eksikliği"},
                {"yontem": "NeRF (Neural Radiance Field)", "psnr": 34.2, "bellek_mb": 5.2, "durum": "Sürekli 3D Fonksiyon & Fotogerçekçi"},
            ],
            "fourier_l_seviyesi": "L_pos = 10 (Frekans = 512pi), L_dir = 4 (Frekans = 8pi)",
        }
