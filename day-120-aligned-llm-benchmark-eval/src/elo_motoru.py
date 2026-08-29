"""
Bradley-Terry Tabanlı Chatbot Arena Elo Derecelendirme Motoru (Day 120 - Faz 6 Capstone).
Modellerin ikili maç sonuçlarına göre dinamik Elo puanı ve kazanma olasılıklarını hesaplar.
"""

from typing import Dict, List, Tuple, Any
import math


class ChatbotArenaEloMotoru:
    """LMSYS Chatbot Arena tarzı dinamik Elo ve kazanma matrisi motoru."""

    def __init__(self, baslangic_elo: float = 1000.0, k_faktoru: float = 32.0):
        self.baslangic_elo = baslangic_elo
        self.k_faktoru = k_faktoru
        self.elo_tablosu: Dict[str, float] = {}
        self.mac_sayilari: Dict[str, int] = {}
        self.galibiyet_sayilari: Dict[str, float] = {}
        self.ikili_sonuclar: Dict[Tuple[str, str], List[float]] = {}

    def _model_kaydet_if_needed(self, model_adi: str):
        if model_adi not in self.elo_tablosu:
            self.elo_tablosu[model_adi] = self.baslangic_elo
            self.mac_sayilari[model_adi] = 0
            self.galibiyet_sayilari[model_adi] = 0.0

    def beklenen_skor(self, r_a: float, r_b: float) -> float:
        """Bradley-Terry beklenen kazanma olasılığı: E_A = 1 / (1 + 10^((R_B - R_A)/400))."""
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def mac_isle(self, model_a: str, model_b: str, sonuc: int):
        """
        Maç sonucunu işler ve Elo puanlarını günceller.
        sonuc: +1 (A Kazandı), 0 (Beraberlik), -1 (B Kazandı)
        """
        self._model_kaydet_if_needed(model_a)
        self._model_kaydet_if_needed(model_b)

        r_a = self.elo_tablosu[model_a]
        r_b = self.elo_tablosu[model_b]

        e_a = self.beklenen_skor(r_a, r_b)
        e_b = self.beklenen_skor(r_b, r_a)

        if sonuc == 1:
            s_a, s_b = 1.0, 0.0
        elif sonuc == -1:
            s_a, s_b = 0.0, 1.0
        else:
            s_a, s_b = 0.5, 0.5

        # Elo Güncellemesi
        self.elo_tablosu[model_a] += self.k_faktoru * (s_a - e_a)
        self.elo_tablosu[model_b] += self.k_faktoru * (s_b - e_b)

        self.mac_sayilari[model_a] += 1
        self.mac_sayilari[model_b] += 1
        self.galibiyet_sayilari[model_a] += s_a
        self.galibiyet_sayilari[model_b] += s_b

        cift_key = (model_a, model_b)
        if cift_key not in self.ikili_sonuclar:
            self.ikili_sonuclar[cift_key] = []
        self.ikili_sonuclar[cift_key].append(s_a)

    def liderlik_tablosu(self) -> List[Dict[str, Any]]:
        """Modelleri Elo puanına göre büyükten küçüğe sıralar."""
        sirali = sorted(self.elo_tablosu.items(), key=lambda x: x[1], reverse=True)
        tablo = []
        for sira, (isim, elo) in enumerate(sirali, start=1):
            mac = self.mac_sayilari[isim]
            galibiyet = self.galibiyet_sayilari[isim]
            kazanma_orani = (galibiyet / max(1, mac)) * 100.0
            tablo.append({
                "sira": sira,
                "model_adi": isim,
                "elo": float(elo),
                "toplam_mac": mac,
                "kazanma_orani": float(kazanma_orani),
            })
        return tablo
