"""
Tümdengelimsel Kıyas ve Sağlamlık Motoru (Day 153 - Faz 8).
Argümanın biçimsel geçerliliğini (Validity) ve sağlamlığını (Soundness) denetleyen ana mantık motoru.
"""

from typing import Dict, Any, List
from .oncul_sonuc_ayristirici import OnculSonucAyristirici
from .safsata_tespitcisi import SafsataTespitcisi


class TumdengelimMotoru:
    """Tümdengelimsel akıl yürütme, kıyas geçerliliği ve safsata denetim motoru."""

    def __init__(self):
        self.ayristirici = OnculSonucAyristirici()
        self.safsata_tespitcisi = SafsataTespitcisi()

    def argumani_degerlendir(
        self,
        arguman_metni: str,
        oncul_dogruluklari: Dict[str, bool] = None,
    ) -> Dict[str, Any]:
        """
        Argümanı ayrıştırır, safsata taraması yapar ve biçimsel geçerlilik/sağlamlık hesaplar.
        """
        ayristirma = self.ayristirici.ayristir(arguman_metni)
        onculler = ayristirma["onculler"]
        sonuc = ayristirma["sonuc"]

        # Safsata denetimi
        safsata_raporu = self.safsata_tespitcisi.safsata_tara(onculler, sonuc)

        # Geçerlilik (Validity) Hesabı
        # Eğer safsata varsa argüman geçersizdir (Invalid)
        gecerli_mi = not safsata_raporu["safsata_var_mi"]

        # Öncüllerin gerçek dünya doğruluğu (True Premises)
        # Varsayılan: Öncüller aksi belirtilmedikçe True kabul edilir
        tum_onculler_dogru_mu = True
        if oncul_dogruluklari:
            tum_onculler_dogru_mu = all(oncul_dogruluklari.values())

        # Sağlamlık (Soundness) = Geçerlilik AND Öncüllerin Doğruluğu
        saglam_mi = gecerli_mi and tum_onculler_dogru_mu

        guven_skoru = 1.0 if saglam_mi else (0.4 if gecerli_mi else 0.0)

        return {
            "ham_arguman": arguman_metni,
            "onculler": onculler,
            "sonuc": sonuc,
            "gecerli_mi": gecerli_mi,
            "saglam_mi": saglam_mi,
            "tum_onculler_dogru_mu": tum_onculler_dogru_mu,
            "safsata_bilgisi": safsata_raporu,
            "guven_skoru": guven_skoru,
            "mantik_formu": (
                "Modus Ponens / Geçerli Kıyas"
                if gecerli_mi
                else safsata_raporu["safsata_adi"]
            ),
        }
