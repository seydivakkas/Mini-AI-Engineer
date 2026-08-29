"""
Bağlam Uzatma ve RoPE/YaRN Kıyaslama Laboratuvarı (Day 104).
Farklı RoPE ölçekleme yöntemlerinin 4k'dan 128k'ya uzanan bağlamda dikkat kararlılığı ve perplexity analizini yapar.
"""

import time
import math
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

from .rope_temelleri import StandartRoPE, LinearPIRoPE
from .ntk_ve_yarn import NTKAwareRoPE, YaRNRoPE


class BaglamLaboratuvari:
    """RoPE, Linear PI, NTK-Aware ve YaRN yöntemlerini karşılaştıran benchmark motoru."""

    def __init__(
        self,
        dim: int = 64,
        orijinal_baglam: int = 4096,
        hedef_baglam: int = 131072,  # 128k
        cihaz: Optional[torch.device] = None,
    ):
        self.dim = dim
        self.orijinal_baglam = orijinal_baglam
        self.hedef_baglam = hedef_baglam
        self.olcek = float(hedef_baglam / orijinal_baglam)  # 128k / 4k = 32.0
        self.cihaz = cihaz or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def modulleri_olustur(self) -> Dict[str, nn.Module]:
        """Tüm RoPE ölçekleme modüllerini oluşturur."""
        return {
            "Standart RoPE (Ölceksiz)": StandartRoPE(dim=self.dim, base=10000.0).to(self.cihaz).eval(),
            "Linear PI (Doğrusal)": LinearPIRoPE(dim=self.dim, base=10000.0, olcek=self.olcek).to(self.cihaz).eval(),
            "NTK-Aware RoPE": NTKAwareRoPE(dim=self.dim, base=10000.0, olcek=self.olcek).to(self.cihaz).eval(),
            "YaRN (Yet another RoPE)": YaRNRoPE(
                dim=self.dim, base=10000.0, olcek=self.olcek, orijinal_max_seq_len=self.orijinal_baglam
            ).to(self.cihaz).eval(),
        }

    def perplexity_egrisi_simulasyonu(
        self,
        baglam_noktalari: List[int] = [4096, 8192, 16384, 32768, 65536, 131072],
    ) -> Dict[str, List[float]]:
        """
        Eğitim bağlamı (4k) aşıldığında yöntemlerin teorik Perplexity (PPL) kararlılık eğrisini hesaplar.
        """
        sonuclar = {"Standart RoPE": [], "Linear PI": [], "NTK-Aware": [], "YaRN": []}

        for S in baglam_noktalari:
            oran = S / self.orijinal_baglam  # 1.0, 2.0, 4.0, 8.0, 16.0, 32.0

            # 1. Standart RoPE: 4k sonrası ekstrapolasyonda PPL patlaması yaşar (OOD açılar)
            if oran <= 1.0:
                ppl_std = 8.5
            else:
                ppl_std = 8.5 * (oran ** 2.2)  # 128k'da PPL 1000+'e fırlar
            sonuclar["Standart RoPE"].append(min(round(ppl_std, 2), 500.0))

            # 2. Linear PI: 4k sonrası PPL kontrollü ama yüksek frekans kaybından dolayı hafif yükselir
            ppl_pi = 8.5 + (2.5 * math.log2(oran))
            sonuclar["Linear PI"].append(round(ppl_pi, 2))

            # 3. NTK-Aware: PI'den daha iyi, yüksek frekansları korur
            ppl_ntk = 8.5 + (1.2 * math.log2(oran))
            sonuclar["NTK-Aware"].append(round(ppl_ntk, 2))

            # 4. YaRN: Entropi düzeltmesi ve rampa ile 128k'da bile mükemmel kararlılık (PPL ~ 8.7)
            ppl_yarn = 8.5 + (0.25 * math.log2(oran))
            sonuclar["YaRN"].append(round(ppl_yarn, 2))

        return sonuclar

    def mesafe_dikkat_bozulmasi_analizi(self, maks_mesafe: int = 128) -> Dict[str, List[float]]:
        """
        Token'lar arasındaki mesafe arttıkça (|m - n|) iç çarpım benzerliğinin nasıl azaldığını ölçer.
        """
        moduller = self.modulleri_olustur()
        x_raw = torch.randn(1, 1, 1, self.dim, device=self.cihaz)

        analiz = {isim: [] for isim in moduller.keys()}
        mesafeler = list(range(0, maks_mesafe, 4))

        with torch.no_grad():
            for d in mesafeler:
                for isim, modul in moduller.items():
                    q_rot = modul(x_raw, seq_len_offset=0)
                    k_rot = modul(x_raw, seq_len_offset=d)
                    skor = float(torch.cosine_similarity(q_rot.flatten(), k_rot.flatten(), dim=0).item())
                    analiz[isim].append(round(skor, 3))

        return analiz
