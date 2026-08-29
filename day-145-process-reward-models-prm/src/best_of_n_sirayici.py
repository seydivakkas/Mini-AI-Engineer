"""
Best-of-N Akıl Yürütme Sıralayıcısı Modülü (Day 145 - Faz 8).
ORM ve PRM ödül modellerine göre N adayın yeniden sıralanması (Re-ranking) ve karşılaştırılması.
"""

from typing import List, Dict, Any
from .orm_odul_modeli import OutcomeRewardModel
from .prm_odul_modeli import ProcessRewardModel


class BestOfNSirayici:
    """N adet düşünce yolunu ORM ve PRM ile puanlayıp en iyi çözümü seçen sıralayıcı."""

    def __init__(self, dogru_cevap: str = "0.05"):
        self.orm = OutcomeRewardModel(dogru_cevap=dogru_cevap)
        self.prm = ProcessRewardModel()

    def karsilastir_ve_sirala(self, aday_yollar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tüm adayları ORM ve PRM ile değerlendirir ve iki modelin seçimlerini karşılaştırır.
        """
        orm_sonuclari = []
        prm_sonuclari = []

        for yol in aday_yollar:
            orm_res = self.orm.puanla(yol)
            prm_res = self.prm.puanla(yol)

            birlesik_orm = {**yol, **orm_res}
            birlesik_prm = {**yol, **prm_res}

            orm_sonuclari.append(birlesik_orm)
            prm_sonuclari.append(birlesik_prm)

        # Sıralamalar
        orm_sirali = sorted(orm_sonuclari, key=lambda x: x["orm_puani"], reverse=True)
        prm_sirali = sorted(prm_sonuclari, key=lambda x: x["prm_carpim_puani"], reverse=True)

        orm_secimi = orm_sirali[0]
        prm_secimi = prm_sirali[0]

        # Şanslı tahmin (Lucky guess) tespit sayısı
        sansli_tahmin_sayisi = sum(
            1 for y in orm_sonuclari if y["nihai_cevap_dogru_mu"] and not self.prm.puanla(y)["gecerli_yol_mu"]
        )

        return {
            "toplam_aday_sayisi": len(aday_yollar),
            "orm_secimi": orm_secimi,
            "prm_secimi": prm_secimi,
            "orm_sirali_adaylar": orm_sirali,
            "prm_sirali_adaylar": prm_sirali,
            "sansli_tahmin_sayisi": sansli_tahmin_sayisi,
            "prm_ustunlugu": prm_secimi.get("gecerli_yol_mu", False) and not orm_secimi.get("ara_hata_tespit_edildi_mi", True),
        }
