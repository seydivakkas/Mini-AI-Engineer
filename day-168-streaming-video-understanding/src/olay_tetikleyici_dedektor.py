"""
Online Olay Tetikleyici Dedektör (Online Event Trigger Detector) Modülü (Day 168 - FAZ 9).
Kareler arası görsel değişim, hareket veya anomali skoru eşiği aştığında VLM çıkarımını tetikler.
"""

from typing import Tuple
import torch
import torch.nn.functional as F


class OlayTetikleyiciDedektor:
    """Anlık değişim tespit edip VLM çıkarımını tetikleyen dedektör."""

    def __init__(self, degisim_esigi: float = 0.35):
        self.degisim_esigi = degisim_esigi
        self.onceki_vektor = None

    def olay_tetiklendi_mi(self, simdiki_vektor: torch.Tensor) -> Tuple[bool, float]:
        """
        simdiki_vektor: [D] boyutunda küresel öznitelik vektörü.
        Döner: (tetiklendi_mi: bool, anomali_skoru: float)
        """
        if self.onceki_vektor is None:
            self.onceki_vektor = simdiki_vektor.clone().detach()
            return False, 0.0

        # Kosinüs mesafesi (1 - Cosine Similarity) ile anomali / değişim skoru
        v1 = self.onceki_vektor.unsqueeze(0)
        v2 = simdiki_vektor.unsqueeze(0)
        benzerlik = F.cosine_similarity(v1, v2).item()
        fark_skoru = max(0.0, float(1.0 - benzerlik))

        tetiklendi = fark_skoru >= self.degisim_esigi
        self.onceki_vektor = simdiki_vektor.clone().detach()
        return tetiklendi, fark_skoru
