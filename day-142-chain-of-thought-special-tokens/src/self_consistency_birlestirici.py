"""
Self-Consistency Birleştirici Modülü (Day 142 - Faz 8).
Çoklu akıl yürütme yollarında çoğunluk oylaması (Majority Voting) ve marjinalizasyon yapan motor.
"""

from typing import List, Dict, Any
from collections import Counter


class SelfConsistencyBirlestirici:
    """Çoklu CoT akıl yürütme yollarını çoğunluk oyuyla birleştiren motor."""

    @classmethod
    def birlestir(cls, orneklenen_yollar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        K adet akıl yürütme yolundan çoğunluk oyu (Majority Vote) ile nihai konsensüs yanıtını seçer.
        """
        if not orneklenen_yollar:
            return {"konsensus_yanit": "", "konsensus_skoru": 0.0, "oy_dagilimi": {}}

        tahminler = [y["tahmin"] for y in orneklenen_yollar]
        toplam_ornek = len(tahminler)

        oy_sayilari = Counter(tahminler)
        kazanan_tahmin, kazanan_oy = oy_sayilari.most_common(1)[0]
        konsensus_skoru = kazanan_oy / toplam_ornek

        # Kazanan tahmine sahip ilk temsilci yolu bul
        temsilci_yol = next(y for y in orneklenen_yollar if y["tahmin"] == kazanan_tahmin)
        sapan_yollar = [y for y in orneklenen_yollar if y["tahmin"] != kazanan_tahmin]

        oy_dagilimi = {k: v for k, v in oy_sayilari.items()}

        return {
            "kazanan_tahmin": kazanan_tahmin,
            "nihai_yanit": temsilci_yol["nihai_yanit"],
            "kazanan_strateji": temsilci_yol["strateji"],
            "temsilci_dusunce_metni": temsilci_yol["dusunce_metni"],
            "toplam_oy": toplam_ornek,
            "kazanan_oy": kazanan_oy,
            "konsensus_skoru": round(konsensus_skoru, 4),
            "oy_dagilimi": oy_dagilimi,
            "sapan_yol_sayisi": len(sapan_yollar),
            "guvenli_mi": konsensus_skoru >= 0.60,
        }
