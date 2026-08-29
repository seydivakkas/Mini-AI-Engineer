"""
Öncül ve Sonuç Ayrıştırıcı Modülü (Day 153 - Faz 8).
Doğal dildeki argümanları mantıksal öncüllere (Premises) ve sonuca (Conclusion) ayrıştırır.
"""

from typing import List, Dict, Any, Tuple


class OnculSonucAyristirici:
    """Argüman metnini analiz ederek öncülleri ve sonucu yapılandırır."""

    SONUC_BAGLACLARI = ["dolayısıyla", "o halde", "bu yüzden", "sonuç olarak", "bu nedenle", "demek ki"]

    @classmethod
    def ayristir(cls, arguman_metni: str) -> Dict[str, Any]:
        """
        Argüman metnini cümlelere bölerek öncülleri ve nihai sonucu ayıklar.
        """
        temiz_metin = arguman_metni.strip()
        cumleler = [c.strip() for c in temiz_metin.replace("\n", ". ").split(".") if c.strip()]

        onculler: List[str] = []
        sonuc: str = ""

        for cumle in cumleler:
            kucuk = cumle.lower()
            sonuc_bulundu = False
            for baglac in cls.SONUC_BAGLACLARI:
                if baglac in kucuk:
                    # Bağlaçtan sonrasını veya tüm cümleyi sonuç al
                    sonuc = cumle
                    sonuc_bulundu = True
                    break
            if not sonuc_bulundu:
                onculler.append(cumle)

        # Eğer bağlaç yoksa son cümleyi sonuç, öncekileri öncül kabul et
        if not sonuc and onculler:
            sonuc = onculler.pop(-1)

        return {
            "ham_arguman": arguman_metni,
            "onculler": onculler,
            "sonuc": sonuc,
            "oncul_sayisi": len(onculler),
        }
