"""
İnteraktif İspat Asistanı (ITP) ve Tip Denetleyici Modülü (Day 152 - Faz 8).
Curry-Howard eşbiçimliliği ve biçimsel ispat doğrulama çekirdeği.
"""

from typing import Dict, Any, List
from .lean4_taktik_motoru import Lean4TaktikMotoru
from .formal_teorem_ureticisi import FormalTeoremUreticisi


class ITPDogrulayici:
    """Lean 4 ispat zincirini adım adım yürüten ve biçimsel olarak doğrulayan ITP çekirdeği."""

    @classmethod
    def teoremi_ispatla_ve_dogrula(cls, dogal_dil_teoremi: str) -> Dict[str, Any]:
        """
        Doğal dil teoremini Lean 4 koduna çevirir ve taktik motorunda doğrulayarak ispatı tamamlar.
        """
        ureticisi = FormalTeoremUreticisi()
        teorem_bilgisi = ureticisi.teoremi_bicimsellestir(dogal_dil_teoremi)

        motor = Lean4TaktikMotoru(
            teorem_adi=teorem_bilgisi["teorem_adi"],
            baslangic_hedefi=teorem_bilgisi["hedef_ifadesi"],
        )

        adim_kayitlari = []
        for taktik in teorem_bilgisi["taktik_adimlari"]:
            kayit = motor.taktik_uygula(taktik)
            adim_kayitlari.append(kayit)

        return {
            "dogal_dil_teoremi": dogal_dil_teoremi,
            "teorem_adi": teorem_bilgisi["teorem_adi"],
            "lean4_kodu": teorem_bilgisi["lean4_kodu"],
            "ispatlandi_mi": motor.ispatlandi_mi,
            "kalan_hedef_sayisi": len(motor.acik_hedefler),
            "adim_kayitlari": adim_kayitlari,
            "taktik_sayisi": len(adim_kayitlari),
        }
