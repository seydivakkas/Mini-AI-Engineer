"""
TDD Döngüsü Yöneticisi Modülü (Day 151 - Faz 8).
Kod Yazma -> Test Koşma -> Hata Ayıklama (Self-Repair) döngü orkestratörü.
"""

from typing import Dict, Any, List
from .kod_ureticisi import KodUreticisi
from .test_yurutucu import TestYurutucu


class TDDDongusuYoneticisi:
    """TDD tabanlı kod üretim ve otomatik hata onarım motoru."""

    def __init__(self, maks_deneme: int = 3):
        self.maks_deneme = maks_deneme
        self.kod_ureticisi = KodUreticisi()
        self.test_yurutucu = TestYurutucu()

    def tdd_dongusunu_baslat(self, gorev_tanimi: str = "Run-Length Encoding Fonksiyonu") -> Dict[str, Any]:
        """
        Uçtan uca TDD döngüsünü çalıştırır.
        """
        dongu_gecmisi: List[Dict[str, Any]] = []

        # 1. İlk Taslak Kod Üretimi
        kod_bilgisi = self.kod_ureticisi.ilk_kodu_uret(gorev_tanimi)

        for tur in range(1, self.maks_deneme + 1):
            kod_metni = kod_bilgisi["kod"]

            # 2. Testleri Koştur
            test_sonucu = self.test_yurutucu.testleri_kostur(kod_metni)

            kayit = {
                "tur": tur,
                "kod": kod_metni,
                "gecen_sayisi": test_sonucu["gecen_sayisi"],
                "toplam_test_sayisi": test_sonucu["toplam_test_sayisi"],
                "basari_orani": test_sonucu["gecen_sayisi"] / test_sonucu["toplam_test_sayisi"],
                "tum_testler_gecti_mi": test_sonucu["tum_testler_gecti_mi"],
                "hata_raporu": test_sonucu["hata_raporu"],
                "ayrintilar": test_sonucu["ayrintilar"],
                "onarma_monologu": kod_bilgisi.get("onarma_monologu", None),
            }
            dongu_gecmisi.append(kayit)

            if test_sonucu["tum_testler_gecti_mi"]:
                # Tüm testler başarıyla geçti
                break

            # 3. Kodu Otomatik Onar (Self-Repair)
            kod_bilgisi = self.kod_ureticisi.kodu_onar(
                gorev=gorev_tanimi,
                onceki_kod=kod_metni,
                hata_raporu=test_sonucu["hata_raporu"] or "Bilinmeyen test hatası",
                tur=tur + 1,
            )

        return {
            "gorev_tanimi": gorev_tanimi,
            "toplam_tur": len(dongu_gecmisi),
            "basarili_mi": dongu_gecmisi[-1]["tum_testler_gecti_mi"],
            "nihai_kod": dongu_gecmisi[-1]["kod"],
            "dongu_gecmisi": dongu_gecmisi,
        }
