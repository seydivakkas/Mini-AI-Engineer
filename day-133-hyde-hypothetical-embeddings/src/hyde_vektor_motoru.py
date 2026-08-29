"""
HyDE Vektör Motoru ve Centroid Birleştirici Modülü (Day 133 - Faz 7).
Hipotez belgelerini vektörleştirip centroid ortalaması alan dense embedding motoru.
"""

from typing import List
import numpy as np
import torch
import torch.nn.functional as F


class HyDEVektorMotoru:
    """Metinleri L2 normalize vektörlere dönüştürür ve HyDE centroid hesaplar."""

    def __init__(self, vektor_boyutu: int = 128):
        self.vektor_boyutu = vektor_boyutu

    def metin_vektorlestir(self, metin: str) -> torch.Tensor:
        """Metin için L2 normalize dense tensör üretir."""
        sozcukler = metin.lower().split()
        vektor = np.zeros(self.vektor_boyutu, dtype=np.float32)

        for idx, kelime in enumerate(sozcukler):
            np.random.seed(abs(hash(kelime)) % (2**31))
            agirlik = 1.0 / (idx + 1) ** 0.4
            vektor += np.random.randn(self.vektor_boyutu).astype(np.float32) * agirlik

        tensör = torch.tensor(vektor, dtype=torch.float32).unsqueeze(0)
        return F.normalize(tensör, p=2, dim=1)

    def toplu_vektorlestir(self, metinler: List[str]) -> torch.Tensor:
        """Metin listesini (B, D) boyutunda tensöre dönüştürür."""
        vektorler = [self.metin_vektorlestir(m) for m in metinler]
        return torch.cat(vektorler, dim=0)

    def hyde_centroid_vektoru_hesapla(self, hipotezler: List[str]) -> torch.Tensor:
        """
        N adet hipotez belgesinin embedding'lerini ortalayarak HyDE centroid vektörünü üretir:
        e_HyDE = Normalize( 1/N * sum(e_i) )
        """
        if not hipotezler:
            raise ValueError("En az 1 hipotez belgesi gereklidir.")

        hipotez_vektorleri = self.toplu_vektorlestir(hipotezler)  # (N, D)
        ortalama_vektor = torch.mean(hipotez_vektorleri, dim=0, keepdim=True)  # (1, D)
        return F.normalize(ortalama_vektor, p=2, dim=1)

    @staticmethod
    def kosinus_benzerligi(vektor_a: torch.Tensor, vektor_b: torch.Tensor) -> float:
        """İki tensör arasındaki kosinüs benzerliğini hesaplar."""
        return float(F.cosine_similarity(vektor_a, vektor_b).squeeze().item())
