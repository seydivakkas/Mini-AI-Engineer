"""
Kirchenbauer LLM Filigran Tespit ve Z-Skoru Hipotez Testi Modülü (Day 118).
Token dizisindeki Yeşil Token oranını hesaplayarak metnin AI tarafından filigranlanıp filigranlanmadığını matematiksel olarak kanıtlar.
"""

from typing import List, Dict, Any, Tuple
import math
from scipy.stats import norm
import hashlib
import torch


class WatermarkDetector:
    """Verilen token dizisini Z-Skoru hipotez testi ile denetleyen filigran tespit motoru."""

    def __init__(
        self,
        vocab_size: int = 1000,
        gamma: float = 0.5,
        gizli_anahtar: int = 15485863,
        z_esigi: float = 4.0,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.gizli_anahtar = gizli_anahtar
        self.z_esigi = z_esigi
        self.green_size = int(vocab_size * gamma)

    def _yesil_listeyi_uret(self, onceki_token: int) -> set:
        tohum_verisi = f"{onceki_token}_{self.gizli_anahtar}".encode("utf-8")
        hash_degeri = int(hashlib.sha256(tohum_verisi).hexdigest(), 16) % (2**32)

        gen = torch.Generator().manual_seed(hash_degeri)
        permutasyon = torch.randperm(self.vocab_size, generator=gen).tolist()
        return set(permutasyon[: self.green_size])

    def filigran_analizi(self, token_dizisi: List[int]) -> Dict[str, Any]:
        """
        Token dizisi üzerinde Yeşil token sayımı yapar ve Z-Skoru hesaplar.
        Z = (|G| - gamma * T) / sqrt(T * gamma * (1 - gamma))
        """
        toplam_gecis = len(token_dizisi) - 1
        if toplam_gecis <= 0:
            return {
                "toplam_token": len(token_dizisi),
                "yesil_token_sayisi": 0,
                "yesil_oran": 0.0,
                "z_skoru": 0.0,
                "p_degeri": 1.0,
                "filigran_var_mi": False,
            }

        yesil_sayisi = 0
        for i in range(toplam_gecis):
            onceki = token_dizisi[i]
            mevcut = token_dizisi[i + 1]
            yesil_liste = self._yesil_listeyi_uret(onceki)
            if mevcut in yesil_liste:
                yesil_sayisi += 1

        beklenen_yesil = self.gamma * toplam_gecis
        standart_sapma = math.sqrt(toplam_gecis * self.gamma * (1.0 - self.gamma))

        z_skoru = (yesil_sayisi - beklenen_yesil) / max(1e-8, standart_sapma)
        p_degeri = float(1.0 - norm.cdf(z_skoru))
        filigran_var_mi = bool(z_skoru >= self.z_esigi)

        return {
            "toplam_token": len(token_dizisi),
            "toplam_gecis": toplam_gecis,
            "yesil_token_sayisi": yesil_sayisi,
            "yesil_oran": yesil_sayisi / toplam_gecis,
            "beklenen_oran": self.gamma,
            "z_skoru": float(z_skoru),
            "p_degeri": p_degeri,
            "filigran_var_mi": filigran_var_mi,
        }
