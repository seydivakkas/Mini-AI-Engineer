"""
Karşıgelişçi Akıl Yürütme (Counterfactual Reasoning) Motoru (Day 159 - Faz 8).
Judea Pearl'ün 3. Basamağını (Abduction -> Action -> Prediction) hesaplar.
"""

from typing import Dict, Any
from .nedensel_dag_modeli import NedenselDAGModeli


class KarsigelisciAkilYurutucu:
    """Bireysel düzeyde karşıgelişçi ('Ya öyle olmasaydı?') analiz motoru."""

    @classmethod
    def karsigelisci_analiz(
        cls,
        model: NedenselDAGModeli,
        birey_z: int = 0,  # 0: Genç
        gerceklesen_x: int = 1,  # İlaç Aldı
        gerceklesen_y: int = 1,  # İyileşti
        karsigelisci_x: int = 0,  # Eğer İlaç Almasaydı?
    ) -> Dict[str, Any]:
        """
        Adım 1: Abduction (Bireysel arka plan Z=birey_z sabitlenir).
        Adım 2: Action (Müdahale X = karsigelisci_x yapılır).
        Adım 3: Prediction (Y'nin yeni olasılığı hesaplanır).
        """
        # P(Y=1 | X=karsigelisci_x, Z=birey_z)
        yeni_iyilesme_olasiligi = model.p_y_given_xz[(karsigelisci_x, birey_z)]

        # Zorunluluk Olasılığı (Probability of Necessity - PN)
        # İyileşmenin ilaca borçlu olma olasılığı
        orijinal_olasilik = model.p_y_given_xz[(gerceklesen_x, birey_z)]
        zorunluluk_orani = (orijinal_olasilik - yeni_iyilesme_olasiligi) / orijinal_olasilik

        return {
            "birey_yas_grubu": "Genç" if birey_z == 0 else "Yaşlı",
            "gerceklesen_tedavi": "İlaç Aldı" if gerceklesen_x == 1 else "İlaç Almadı",
            "gerceklesen_sonuc": "İyileşti" if gerceklesen_y == 1 else "İyileşmedi",
            "karsigelisci_eylem": "İlaç Almasaydı",
            "karsigelisci_iyilesme_olasiligi": round(yeni_iyilesme_olasiligi, 3),  # 0.800 (%80)
            "zorunluluk_olasiligi_pn": round(zorunluluk_orani, 3),  # 0.111 (%11.1)
        }
