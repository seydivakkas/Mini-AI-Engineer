"""
Self-Play RL Profilleyici ve Simülasyon Modülü (Day 210 - FAZ 11).
100 Turluk Dinamik Zorluk Müfredatı, Yetenek Büyümesi ve Sentetik Veri Hacmi.
"""

from typing import Dict, Any, List
from .self_play_motoru import SelfPlayRLTrainer


class SelfPlayProfilleyici:
    """Self-Play RL Müfredat Profilleyicisi."""

    @classmethod
    def simulasyon_yurut(cls, toplam_tur: int = 100) -> Dict[str, Any]:
        """100 turluk kendi kendine öğrenme simülasyonunu çalıştırır."""
        trainer = SelfPlayRLTrainer(baslangic_yetenek=1.5, ogrenme_hizi=0.08)

        turlar = []
        zorluklar = []
        yetenekler = []
        solver_odulleri = []
        generator_odulleri = []
        dogru_sayisi = 0

        for t in range(1, toplam_tur + 1):
            adim = trainer.adim_yurut()

            turlar.append(t)
            zorluklar.append(round(adim["yeni_zorluk"], 2))
            yetenekler.append(round(adim["guncel_yetenek"], 2))
            solver_odulleri.append(adim["oduller"]["r_solver"])
            generator_odulleri.append(round(adim["oduller"]["r_generator"], 2))

            if adim["oduller"]["dogru_mu"]:
                dogru_sayisi += 1

        genel_dogruluk = (dogru_sayisi / toplam_tur) * 100.0

        return {
            "toplam_tur": toplam_tur,
            "turlar": turlar,
            "zorluk_egrisi": zorluklar,
            "yetenek_egrisi": yetenekler,
            "solver_odulleri": solver_odulleri,
            "generator_odulleri": generator_odulleri,
            "genel_dogruluk_orani": genel_dogruluk,
            "baslangic_zorluk": zorluklar[0],
            "son_zorluk": zorluklar[-1],
            "baslangic_yetenek": yetenekler[0],
            "son_yetenek": yetenekler[-1],
        }
