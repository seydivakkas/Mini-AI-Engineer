"""
Monte Carlo Tree Search (MCTS) Düğüm Veri Yapısı (Day 146 - Faz 8).
UCT (Upper Confidence bounds for Trees) skorlama ve ziyaret istatistikleri.
"""

from typing import List, Optional, Dict, Any
import math


class MCTSDugumu:
    """MCTS arama ağacındaki her bir düşünce düğümü."""

    def __init__(
        self,
        durum_id: str,
        sayilar: List[float],
        adim_gecmisi: List[str] = None,
        ebeveyn: Optional["MCTSDugumu"] = None,
    ):
        self.durum_id = durum_id
        self.sayilar = [float(s) for s in sayilar]
        self.adim_gecmisi = adim_gecmisi or []
        self.ebeveyn = ebeveyn
        self.cocuklar: List["MCTSDugumu"] = []

        # MCTS İstatistikleri
        self.ziyaret_sayisi: int = 0  # N(s)
        self.toplam_odul: float = 0.0  # W(s)

    @property
    def ortalama_deger(self) -> float:
        """Q(s) = W(s) / N(s)"""
        if self.ziyaret_sayisi == 0:
            return 0.0
        return self.toplam_odul / self.ziyaret_sayisi

    def uct_skoru(self, c_kesif: float = 1.414) -> float:
        """
        Upper Confidence bounds for Trees (UCT) formülü:
        UCT = Q(s) + c * sqrt( ln(N(ebeveyn)) / (N(s) + 1e-6) )
        """
        if self.ziyaret_sayisi == 0:
            return float("inf")  # Henüz keşfedilmemiş düğümlere sonsuz öncelik

        ebeveyn_ziyaret = self.ebeveyn.ziyaret_sayisi if self.ebeveyn else self.ziyaret_sayisi
        kesif_terimi = c_kesif * math.sqrt(math.log(max(1, ebeveyn_ziyaret)) / self.ziyaret_sayisi)
        return self.ortalama_deger + kesif_terimi

    def en_iyi_cocugu_sec(self, c_kesif: float = 1.414) -> "MCTSDugumu":
        """En yüksek UCT skoruna sahip çocuğu seçer (Selection adımı)."""
        return max(self.cocuklar, key=lambda c: c.uct_skoru(c_kesif))

    def en_cok_ziyaret_edilen_cocuk(self) -> "MCTSDugumu":
        """Nihai karar anında en çok ziyaret edilen (en sağlam) çocuğu seçer."""
        return max(self.cocuklar, key=lambda c: c.ziyaret_sayisi)

    def terminal_mi(self) -> bool:
        """Kalan tek sayı varsa terminal durumdur."""
        return len(self.sayilar) <= 1

    def to_dict(self) -> Dict[str, Any]:
        """Serileştirilmiş durum sözlüğü."""
        return {
            "durum_id": self.durum_id,
            "sayilar": self.sayilar,
            "adim_gecmisi": self.adim_gecmisi,
            "ziyaret_sayisi": self.ziyaret_sayisi,
            "ortalama_q": round(self.ortalama_deger, 4),
            "toplam_w": round(self.toplam_odul, 4),
        }
