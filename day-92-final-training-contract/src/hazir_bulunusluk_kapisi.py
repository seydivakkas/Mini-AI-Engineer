"""
Day 92: Eğitim Hazır Bulunuşluk Karar Kapısı (Training Readiness Gate)
---------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .sozlesme_kurallari import IhlalSeviyesi, KuralIhlali
from .veri_denetleyici import DenetimSonucu
from .sizinti_dedektoru import SizintiRaporu


class KapiDurumu(Enum):
    ONAYLANDI = "ONAYLANDI (PASS)"
    UYARI_ILE_ONAY = "UYARI İLE ONAY (PASS WITH WARNING)"
    BLOKE_EDILDI = "BLOKE EDİLDİ (FAIL / BLOCKED)"


@dataclass
class KapiKarari:
    durum: KapiDurumu
    egitim_baslatilabilir_mi: bool
    toplam_ihlal_sayisi: int
    bloke_eden_hatalar: List[str]
    uyarilar: List[str]
    ozet_mesaj: str


class HazirBulunuslukKapisi:
    """
    Tüm veri sözleşmesi ve sızıntı denetimlerini birleştirip GPU eğitim hattının
    başlatılmasına onay veren veya eğitimi derhal durduran güvenlik kapısı.
    """

    def __init__(self, sızıntıda_bloke_et: bool = True):
        self.sızıntıda_bloke_et = sızıntıda_bloke_et

    def degerlendir(
        self,
        denetim_sonucu: DenetimSonucu,
        sizinti_raporu: Optional[SizintiRaporu] = None,
    ) -> KapiKarari:
        bloke_hatalari: List[str] = []
        uyarilar: List[str] = []

        for ihlal in denetim_sonucu.ihlal_listesi:
            if ihlal.seviye == IhlalSeviyesi.BLOKE_EDICI:
                bloke_hatalari.append(f"[{ihlal.kural_adi}] {ihlal.mesaj}")
            elif ihlal.seviye == IhlalSeviyesi.UYARI:
                uyarilar.append(f"[{ihlal.kural_adi}] {ihlal.mesaj}")

        # Sızıntı kontrolü
        if sizinti_raporu and sizinti_raporu.sizinti_var_mi:
            mesaj = f"[VERI_SIZINTISI] Train ve Val arasında {sizinti_raporu.kesisen_ornek_sayisi} adet birebir örtüşen örnek tespit edildi (%{sizinti_raporu.sizinti_orani_val * 100:.2f})!"
            if self.sızıntıda_bloke_et:
                bloke_hatalari.append(mesaj)
            else:
                uyarilar.append(mesaj)

        toplam_ihlal = len(bloke_hatalari) + len(uyarilar)

        if len(bloke_hatalari) > 0:
            durum = KapiDurumu.BLOKE_EDILDI
            baslatilabilir = False
            ozet = f"Eğitim BAŞLATILAMAZ! {len(bloke_hatalari)} adet kritik sözleşme ihlali var."
        elif len(uyarilar) > 0:
            durum = KapiDurumu.UYARI_ILE_ONAY
            baslatilabilir = True
            ozet = f"Eğitim başlatılabilir ancak {len(uyarilar)} adet uyarı mevcut."
        else:
            durum = KapiDurumu.ONAYLANDI
            baslatilabilir = True
            ozet = "Tüm veri sözleşmesi testleri %100 BAŞARILI. Eğitim güvenle başlatılabilir."

        return KapiKarari(
            durum=durum,
            egitim_baslatilabilir_mi=baslatilabilir,
            toplam_ihlal_sayisi=toplam_ihlal,
            bloke_eden_hatalar=bloke_hatalari,
            uyarilar=uyarilar,
            ozet_mesaj=ozet,
        )
