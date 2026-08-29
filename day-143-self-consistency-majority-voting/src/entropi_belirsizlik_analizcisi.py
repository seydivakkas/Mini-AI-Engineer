"""
Entropi ve Epistemik Belirsizlik Analizcisi Modülü (Day 143 - Faz 8).
Shannon Entropisi (H(Y|x)), Gini Saflığı ve Güven Kalibrasyonu motoru.
"""

from typing import Dict, Any
import math


class EntropiBelirsizlikAnalizcisi:
    """Oylama dağılımının Shannon Entropisini ve epistemik belirsizliğini hesaplar."""

    @classmethod
    def analiz_et(cls, olasilik_dagilimi: Dict[str, float]) -> Dict[str, Any]:
        """
        Shannon Entropisi: H(Y|x) = - sum(p * log2(p))
        Gini Kirliliği: 1 - sum(p^2)
        """
        if not olasilik_dagilimi:
            return {"shannon_entropisi": 0.0, "gini_kirliligi": 0.0, "belirsizlik_seviyesi": "Bilinmiyor"}

        # Normalize sağlama
        toplam = sum(olasilik_dagilimi.values())
        norm_p = {k: (v / max(1e-6, toplam)) for k, v in olasilik_dagilimi.items()}

        shannon_entropi = 0.0
        kare_toplam = 0.0

        for p in norm_p.values():
            if p > 1e-9:
                shannon_entropi -= p * math.log2(p)
                kare_toplam += (p ** 2)

        gini_kirliligi = 1.0 - kare_toplam

        # Belirsizlik seviyesi tespiti
        if shannon_entropi < 0.50:
            seviye = "DÜŞÜK_BELİRSİZLİK (YÜKSEK GÜVEN)"
            guvenli = True
        elif shannon_entropi < 1.20:
            seviye = "ORTA_BELİRSİZLİK (KONTROL GEREKLİ)"
            guvenli = True
        else:
            seviye = "YÜKSEK_BELİRSİZLİK (HALÜSİNASYON RİSKİ)"
            guvenli = False

        return {
            "shannon_entropisi": round(shannon_entropi, 4),
            "gini_kirliligi": round(gini_kirliligi, 4),
            "belirsizlik_seviyesi": seviye,
            "guvenli_mi": guvenli,
            "maksimum_olasilik": round(max(norm_p.values()), 4),
        }
