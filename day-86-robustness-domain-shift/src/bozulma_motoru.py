"""
Görsel Bozulmalar (Common Corruptions) ve Dağılım Kayması Motoru
---------------------------------------------------------------
Hendrycks & Dietterich (ICLR 2019) "Benchmarking Neural Network Robustness to Common Corruptions"
makalesinde tanımlanan 8 temel bozulma tipi ve 5 şiddet seviyesini (Severity 1..5) uygulayan motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple, Callable
import torch
import torch.nn.functional as F
import numpy as np


class GorselBozulmaMotoru:
    """
    Görsellere 8 farklı bozulma tipini 5 farklı şiddet derecesinde (1..5) uygulayan simülatör.
    """
    @staticmethod
    def gaussian_noise(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5, "Şiddet 1 ile 5 arasında olmalıdır!"
        std_listesi = [0.10, 0.20, 0.35, 0.50, 0.70]
        std = std_listesi[siddet - 1]
        gurultu = torch.randn_like(x) * std
        return x + gurultu

    @staticmethod
    def tuz_biber_noise(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5
        oranlar = [0.03, 0.07, 0.12, 0.18, 0.25]
        prob = oranlar[siddet - 1]
        maske = torch.rand_like(x)
        x_out = x.clone()
        x_out[maske < (prob / 2.0)] = -2.5  # Tuz (Siyah/Koyu)
        x_out[(maske >= (prob / 2.0)) & (maske < prob)] = 2.5  # Biber (Beyaz/Açık)
        return x_out

    @staticmethod
    def gaussian_blur(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5
        sigmalar = [0.6, 1.0, 1.5, 2.2, 3.0]
        sigma = sigmalar[siddet - 1]
        k_size = 2 * int(4 * sigma + 0.5) + 1
        
        # 1D Gauss çekirdeği
        ax = torch.arange(-k_size // 2 + 1., k_size // 2 + 1.)
        xx = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_1d = xx / xx.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0).to(x.device)
        kernel_2d = kernel_2d.repeat(x.size(1), 1, 1, 1)

        padding = k_size // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=x.size(1))

    @staticmethod
    def motion_blur(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5
        k_sizes = [3, 5, 7, 9, 11]
        k = k_sizes[siddet - 1]
        # Yatay hareket çekirdeği
        kernel_2d = torch.zeros(1, 1, k, k, device=x.device)
        kernel_2d[0, 0, k // 2, :] = 1.0 / k
        kernel_2d = kernel_2d.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, kernel_2d, padding=k // 2, groups=x.size(1))

    @staticmethod
    def parlaklik(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5
        kaymalar = [0.15, 0.30, 0.50, 0.75, 1.05]
        return x + kaymalar[siddet - 1]

    @staticmethod
    def kontrast(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5
        olcekler = [0.80, 0.65, 0.50, 0.35, 0.20]
        faktör = olcekler[siddet - 1]
        ortalama = x.mean(dim=(-2, -1), keepdim=True)
        return (x - ortalama) * faktör + ortalama

    @staticmethod
    def pikselleme(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5
        olcekler = [0.75, 0.55, 0.40, 0.25, 0.15]
        f = olcekler[siddet - 1]
        h, w = x.shape[-2:]
        down_h, down_w = max(2, int(h * f)), max(2, int(w * f))
        kucuk = F.interpolate(x, size=(down_h, down_w), mode="nearest")
        return F.interpolate(kucuk, size=(h, w), mode="nearest")

    @staticmethod
    def jpeg_sikistirma_sim(x: torch.Tensor, siddet: int = 1) -> torch.Tensor:
        assert 1 <= siddet <= 5
        kuantizasyon_adim = [0.2, 0.4, 0.7, 1.1, 1.6][siddet - 1]
        # Yüksek frekans kuantizasyon simülasyonu
        gurultu = torch.round(x / kuantizasyon_adim) * kuantizasyon_adim
        return 0.7 * gurultu + 0.3 * x

    @classmethod
    def tum_bozulma_fonksiyonlari(cls) -> Dict[str, Callable]:
        return {
            "Gaussian Gürültüsü": cls.gaussian_noise,
            "Tuz & Biber Gürültüsü": cls.tuz_biber_noise,
            "Gauss Bulanıklığı": cls.gaussian_blur,
            "Hareket Bulanıklığı": cls.motion_blur,
            "Aşırı Parlaklık": cls.parlaklik,
            "Düşük Kontrast": cls.kontrast,
            "Pikselleme (Düşük Çözünürlük)": cls.pikselleme,
            "Sıkıştırma Bozulması": cls.jpeg_sikistirma_sim,
        }
