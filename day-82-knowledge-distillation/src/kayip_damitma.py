"""
Bilgi Damıtma (Knowledge Distillation) Kayıp Fonksiyonu
-------------------------------------------------------
Hinton et al. (2015) "Distilling the Knowledge in a Neural Network" formülasyonu.
Sert etiket Cross-Entropy kaybı ile Öğretmen modelin sıcaklık (Temperature τ) ile
yumuşatılmış çıktılarının KL-Diverjansını birleştiren kayıp motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilgiDamitmaKaybi(nn.Module):
    """
    Knowledge Distillation Loss:
    L = (1 - α) * CE(z_s, y) + α * τ² * KL( Softmax(z_s / τ) || Softmax(z_t / τ) )
    """
    def __init__(self, sicaklik: float = 4.0, alfa: float = 0.7):
        super().__init__()
        assert sicaklik > 0.0, "Sıcaklık (Temperature τ) pozitif olmalıdır!"
        assert 0.0 <= alfa <= 1.0, "Alfa katsayısı [0, 1] aralığında olmalıdır!"

        self.sicaklik = sicaklik
        self.alfa = alfa
        self.kl_kayip_fn = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        ogrenci_logitleri: torch.Tensor,
        ogretmen_logitleri: torch.Tensor,
        gercek_etiketler: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Girdi:
          ogrenci_logitleri: (Batch, Num_Classes)
          ogretmen_logitleri: (Batch, Num_Classes)
          gercek_etiketler: (Batch,) tamsayı etiketler
        Çıktı:
          toplam_kayip (Tensor), metrikler (Dict)
        """
        tau = self.sicaklik

        # 1. Sert Hedef Kaybı (Hard Label Cross-Entropy)
        ce_kaybi = F.cross_entropy(ogrenci_logitleri, gercek_etiketler)

        # 2. Yumuşak Hedef Kaybı (Soft Target KL-Divergence)
        # Öğrenci: log_softmax(z_s / τ)
        # Öğretmen: softmax(z_t / τ)
        ogrenci_yumusak_log = F.log_softmax(ogrenci_logitleri / tau, dim=-1)
        ogretmen_yumusak_prob = F.softmax(ogretmen_logitleri / tau, dim=-1)

        # KL-Divergence hesapla ve τ² ile ölçekle
        kl_kaybi = self.kl_kayip_fn(ogrenci_yumusak_log, ogretmen_yumusak_prob) * (tau ** 2)

        # 3. Ağırlıklı Toplam Kayıp
        toplam_kayip = (1.0 - self.alfa) * ce_kaybi + self.alfa * kl_kaybi

        metrikler = {
            "toplam_kayip": toplam_kayip.item(),
            "ce_kaybi": ce_kaybi.item(),
            "kl_kaybi": kl_kaybi.item()
        }

        return toplam_kayip, metrikler
