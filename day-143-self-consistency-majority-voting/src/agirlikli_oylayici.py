"""
Ağırlıklı Oylayıcı (Weighted Majority Voting) Modülü (Day 143 - Faz 8).
Düz sayım (Hard Voting) yerine yol olasılıkları (P(trajectory)) ile ağırlıklandırılmış çoğunluk oyu motoru.
"""

from typing import List, Dict, Any
from collections import defaultdict


class AgirlikliOylayici:
    """Yol güven skoru ve log-olasılıkları kullanarak ağırlıklı Self-Consistency marjinalizasyonu yapar."""

    @classmethod
    def oyla(cls, orneklenen_yollar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Hard Voting ve Soft Weighted Voting sonuçlarını karşılaştırmalı olarak hesaplar.
        """
        if not orneklenen_yollar:
            return {"kazanan_tahmin": "", "agirlikli_skor": 0.0, "dagilim": {}}

        hard_oylar: Dict[str, int] = defaultdict(int)
        agirlikli_oylar: Dict[str, float] = defaultdict(float)

        toplam_agirlik = 0.0
        for y in orneklenen_yollar:
            tahmin = y["tahmin"]
            agirlik = y.get("yol_olasiligi", 1.0)

            hard_oylar[tahmin] += 1
            agirlikli_oylar[tahmin] += agirlik
            toplam_agirlik += agirlik

        # Normalize edilmiş ağırlıklı olasılık dağılımı
        normalize_agirlikli: Dict[str, float] = {}
        for k, v in agirlikli_oylar.items():
            normalize_agirlikli[k] = round(v / max(1e-6, toplam_agirlik), 4)

        # Kazananları belirle
        kazanan_hard = max(hard_oylar.items(), key=lambda x: x[1])[0]
        kazanan_weighted = max(normalize_agirlikli.items(), key=lambda x: x[1])[0]

        return {
            "kazanan_tahmin": kazanan_weighted,
            "kazanan_hard_tahmin": kazanan_hard,
            "agirlikli_guven_skoru": normalize_agirlikli[kazanan_weighted],
            "hard_oy_dagilimi": dict(hard_oylar),
            "agirlikli_oy_dagilimi": normalize_agirlikli,
            "toplam_ornek": len(orneklenen_yollar),
        }
