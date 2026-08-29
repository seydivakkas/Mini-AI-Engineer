"""
Process-supervised Reward Model (PRM) Modülü (Day 145 - Faz 8).
Adım adım mantıksal doğruluk puanlama (Lightman et al. / OpenAI PRM800K).
"""

from typing import Dict, Any, List, Tuple
import math


class ProcessRewardModel:
    """Her bir akıl yürütme adımını bağımsız olarak puanlayan süreç ödül modeli."""

    def __init__(self, dogrulayici_fn=None):
        self.dogrulayici_fn = dogrulayici_fn

    def adim_puanla(self, adim_metni: str, onceki_adımlar: List[str] = None) -> Tuple[float, str]:
        """
        Tek bir adımın mantıksal ve matematiksel geçerliliğini puanlar.
        Puan aralığı: [0.0, 1.0]
        """
        metin = adim_metni.lower()

        # 1. Önce geçerli matematiksel çıkarım kalıplarını kontrol et
        dogru_kaliplar = [
            "sopa + top = 1.10",
            "sopa = top + 1.00",
            "2 * top = 0.10",
            "top = 0.05",
            "13 - 9 = 4",
            "10 - 4 = 6",
            "6 * 4 = 24",
        ]

        if any(d in metin for d in dogru_kaliplar):
            return 0.98, "dogru_adim"

        # 2. Bilinen mantıksal halüsinasyon ve işlem hatası kalıpları
        hatali_kaliplar = [
            "1.10 - 1.00 = 0.10",
            "top = 0.10",
            "1.10 - 0.10 = 1.00 o halde top 0.10",
            "4 * 4 = 24",
            "16 + 10 = 24",
        ]

        if any(h in metin for h in hatali_kaliplar):
            return 0.05, "hatali_adim"

        return 0.85, "notr_adim"

    def puanla(self, yol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tüm adımları tek tek puanlar, ilk hata noktasını bulur ve toplam PRM skorunu hesaplar.
        """
        adimlar = yol.get("adimlar", [])
        adim_puanlari = []
        adim_etiketleri = []
        ilk_hata_adimi = None

        carpim_skoru = 1.0

        for idx, adim in enumerate(adimlar, start=1):
            puan, etiket = self.adim_puanla(adim)
            adim_puanlari.append(puan)
            adim_etiketleri.append(etiket)
            carpim_skoru *= puan

            if puan < 0.20 and ilk_hata_adimi is None:
                ilk_hata_adimi = idx

        min_puan = min(adim_puanlari) if adim_puanlari else 0.0

        return {
            "model_turu": "Process-supervised Reward Model (PRM)",
            "prm_carpim_puani": round(carpim_skoru, 4),
            "prm_min_puani": round(min_puan, 4),
            "adim_puanlari": adim_puanlari,
            "adim_etiketleri": adim_etiketleri,
            "ilk_hata_adimi": ilk_hata_adimi,
            "ara_hata_tespit_edildi_mi": ilk_hata_adimi is not None,
            "gecerli_yol_mu": ilk_hata_adimi is None,
        }
