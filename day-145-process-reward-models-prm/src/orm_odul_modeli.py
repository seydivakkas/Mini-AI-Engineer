"""
Outcome-supervised Reward Model (ORM) Modülü (Day 145 - Faz 8).
Yalnızca nihai cevabın doğruluğuna bakan geleneksel sonuç ödül modeli.
"""

from typing import Dict, Any


class OutcomeRewardModel:
    """Yalnızca son cevabın doğruluğuna göre ödül üreten model."""

    def __init__(self, dogru_cevap: str = "0.05"):
        self.dogru_cevap = str(dogru_cevap).strip().lower()

    def puanla(self, yol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Düşünce yolunun sadece nihai cevabını kontrol eder.
        Ara adımlardaki mantık hatalarını göremez!
        """
        nihai_cevap = str(yol.get("nihai_cevap", "")).strip().lower()
        dogru_mu = (nihai_cevap == self.dogru_cevap)

        # ORM nihai doğruysa 1.0 (veya yüksek olasılık), yanlışsa 0.0 verir
        orm_puani = 1.0 if dogru_mu else 0.0

        return {
            "model_turu": "Outcome-supervised Reward Model (ORM)",
            "orm_puani": orm_puani,
            "nihai_cevap_dogru_mu": dogru_mu,
            "ara_hata_tespit_edildi_mi": False,  # ORM yapısal olarak ara hatayı tespit edemez
            "aciklama": "Doğru sonuç" if dogru_mu else "Yanlış sonuç",
        }
