"""
Vision CNN Modeli ve Dayanıklı Eğitim (Robust Training) Motoru
-------------------------------------------------------------
Standart eğitim ile Bozulma-Dayanıklı (Perturbation-Robust / AugMix) eğitimi
kıyaslamalı koşturan derin görme modeli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, List, Optional
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .bozulma_motoru import GorselBozulmaMotoru


class DayanikliVisionModeli(nn.Module):
    """
    3 Aşamalı Konvolüsyonel Görme Modeli
    """
    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 32):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi

        self.omurga = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_kanal, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(),
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(),
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.kafa = nn.Linear(taban_kanal * 4, sinif_sayisi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.omurga(x).flatten(1)
        return self.kafa(x)

    @classmethod
    def egit(
        cls,
        model: nn.Module,
        loader: DataLoader,
        epok_sayisi: int = 15,
        lr: float = 2e-3,
        dayanikli_egitim: bool = False,
        cihaz: str = "cpu"
    ) -> List[float]:
        """
        Modeli standart veya bozulma-dayanıklı (AugMix / Perturbation Training) rejimde eğitir.
        """
        model = model.to(cihaz).train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        kayip_gecmisi = []
        bozulma_fonksiyonlari = list(GorselBozulmaMotoru.tum_bozulma_fonksiyonlari().values())

        for _ in range(epok_sayisi):
            model.train()
            toplam_loss = 0.0
            toplam_ornek = 0

            for x, y in loader:
                x, y = x.to(cihaz), y.to(cihaz)
                opt.zero_grad()

                if dayanikli_egitim:
                    # Temiz görüntü kaybı
                    cikis_temiz = model(x)
                    loss_temiz = F.cross_entropy(cikis_temiz, y)

                    # Bozulmuş görüntü (Hafif-Orta Şiddet s=1..2)
                    secilen_fn = random.choice(bozulma_fonksiyonlari)
                    s = random.choice([1, 2])
                    x_bozuk = secilen_fn(x, siddet=s)
                    cikis_bozuk = model(x_bozuk)
                    loss_bozuk = F.cross_entropy(cikis_bozuk, y)

                    # AugMix tutarlılık kaybı (Temiz ve Bozuk tahminleri yakınlaştırma)
                    p_temiz = F.softmax(cikis_temiz, dim=-1)
                    p_bozuk = F.softmax(cikis_bozuk, dim=-1)
                    p_ort = 0.5 * (p_temiz + p_bozuk)
                    jsd_loss = 0.5 * (F.kl_div(p_temiz.log(), p_ort, reduction="batchmean") +
                                      F.kl_div(p_bozuk.log(), p_ort, reduction="batchmean"))

                    loss = 0.5 * loss_temiz + 0.5 * loss_bozuk + 1.0 * jsd_loss
                else:
                    cikis = model(x)
                    loss = F.cross_entropy(cikis, y)

                loss.backward()
                opt.step()

                toplam_loss += loss.item() * x.size(0)
                toplam_ornek += x.size(0)

            kayip_gecmisi.append(toplam_loss / max(1, toplam_ornek))

        return kayip_gecmisi
