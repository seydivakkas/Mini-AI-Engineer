"""
Diffusion Policy (Difüzyon Tabanlı Robotik Manipülasyon) Motoru (Day 242).
DDPM Gürültü Zamanlayıcı, 1D Zamansal U-Net ve Eylem Bloklama (Action Chunking).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionPolicyScheduler:
    """Robotik Eylemler için DDPM Doğrusal Difüzyon Zamanlayıcısı."""

    def __init__(self, adim_sayisi: int = 16, beta_baslangic: float = 1e-4, beta_bitis: float = 0.02):
        self.adim_sayisi = adim_sayisi
        self.betalar = torch.linspace(beta_baslangic, beta_bitis, adim_sayisi)
        self.alfalar = 1.0 - self.betalar
        self.kumulatif_alfalar = torch.cumprod(self.alfalar, dim=0)

    def gurultu_ekle(self, eylem_0: torch.Tensor, gurultu: torch.Tensor, zaman_adimi: torch.Tensor) -> torch.Tensor:
        """İleri difüzyon: q(A_k | A_0) formülüyle gürültülü eylem bloku üretir."""
        sqrt_alpha = torch.sqrt(self.kumulatif_alfalar[zaman_adimi]).view(-1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - self.kumulatif_alfalar[zaman_adimi]).view(-1, 1, 1)
        return sqrt_alpha * eylem_0 + sqrt_one_minus_alpha * gurultu

    def gurultuden_arindir_adimi(self, tahmin_edilen_gurultu: torch.Tensor, zaman_adimi: int, x_k: torch.Tensor) -> torch.Tensor:
        """Ters difüzyon: p_theta(A_{k-1} | A_k, c) adımı."""
        beta_k = self.betalar[zaman_adimi]
        alpha_k = self.alfalar[zaman_adimi]
        kumulatif_alpha_k = self.kumulatif_alfalar[zaman_adimi]

        katsayi = beta_k / torch.sqrt(1.0 - kumulatif_alpha_k)
        x_onceki = (x_k - katsayi * tahmin_edilen_gurultu) / torch.sqrt(alpha_k)

        if zaman_adimi > 0:
            varyans_gurultusu = torch.randn_like(x_k)
            sigma = torch.sqrt(beta_k)
            x_onceki += sigma * varyans_gurultusu * 0.1

        return x_onceki


class DiffusionUNet1D(nn.Module):
    """1D Zamansal Konvolüsyonel Eylem Tahmin Ağı."""

    def __init__(self, eylem_boyutu: int = 7, eylem_ufku: int = 8, kosul_boyutu: int = 64, gizli_boyut: int = 128):
        super().__init__()
        self.eylem_boyutu = eylem_boyutu
        self.eylem_ufku = eylem_ufku
        self.gizli_boyut = gizli_boyut

        # Zaman Adımı Gömme
        self.zaman_gomme = nn.Sequential(
            nn.Linear(1, gizli_boyut),
            nn.Mish(),
            nn.Linear(gizli_boyut, gizli_boyut),
        )

        # Koşul (Görsel + Durum) Projeksiyonu
        self.kosul_proj = nn.Linear(kosul_boyutu, gizli_boyut)

        # 1D Konvolüsyonel Kodlayıcı-Kod Çözücü Blokları
        self.giris_conv = nn.Conv1d(eylem_boyutu, gizli_boyut, kernel_size=3, padding=1)
        self.orta_blok = nn.Sequential(
            nn.Conv1d(gizli_boyut, gizli_boyut, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(gizli_boyut, gizli_boyut, kernel_size=3, padding=1),
        )
        self.cikis_conv = nn.Conv1d(gizli_boyut, eylem_boyutu, kernel_size=3, padding=1)

    def forward(self, eylem_k: torch.Tensor, zaman_adimi: torch.Tensor, kosul: torch.Tensor) -> torch.Tensor:
        """Gürültü tahmini üretir. Girdi: [B, Ta, Da] -> Çıktı: [B, Ta, Da]"""
        # [B, Ta, Da] -> [B, Da, Ta] (Conv1d formatı)
        x = eylem_k.transpose(1, 2)
        h = self.giris_conv(x)

        # Zaman ve Koşul Katkısı
        z_t = self.zaman_gomme(zaman_adimi.float().view(-1, 1)).unsqueeze(-1)
        c_t = self.kosul_proj(kosul).unsqueeze(-1)
        h = h + z_t + c_t

        h = self.orta_blok(h)
        cikis = self.cikis_conv(h)
        return cikis.transpose(1, 2)


class DiffusionPolicyController:
    """Kayan Ufuklu Kapalı Döngü Robotik Yörünge Kontrolcüsü."""

    def __init__(self, model: DiffusionUNet1D, scheduler: DiffusionPolicyScheduler):
        self.model = model
        self.scheduler = scheduler

    def eylem_bloku_uret(self, kosul: torch.Tensor) -> np.ndarray:
        """DDPM ters difüzyon döngüsüyle Ta adımlı temiz eylem bloku üretir."""
        self.model.eval()
        batch_size = kosul.shape[0]
        eylem_ufku = self.model.eylem_ufku
        eylem_boyutu = self.model.eylem_boyutu

        with torch.no_grad():
            # Başlangıç Saf Gauss Gürültüsü A_K ~ N(0, I)
            x_k = torch.randn(batch_size, eylem_ufku, eylem_boyutu)

            for t in reversed(range(self.scheduler.adim_sayisi)):
                zaman_tensor = torch.tensor([t] * batch_size, dtype=torch.long)
                tahmin_gurultu = self.model(x_k, zaman_tensor, kosul)
                x_k = self.scheduler.gurultuden_arindir_adimi(tahmin_gurultu, t, x_k)

            return x_k.cpu().numpy()[0]

    def kayan_ufuk_icra_et(self, kosul: torch.Tensor, icra_adimi: int = 4) -> np.ndarray:
        """Kayan ufuklu (Receding Horizon) kontrol: Üretilen bloktan ilk Te adımı seçer."""
        tam_blok = self.eylem_bloku_uret(kosul)
        return tam_blok[:icra_adimi]
