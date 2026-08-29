"""
Test Yürütücü ve Hata Yakalama Modülü (Day 151 - Faz 8).
Üretilen Python kodunu birim testlerle çalıştıran ve hata yığını (Traceback) raporlayan motor.
"""

from typing import Dict, Any, Tuple
import traceback


class TestYurutucu:
    """Üretilen fonksiyonları birim test paketiyle sınayan izole test yürütücü."""

    __test__ = False  # PyTest'in test sınıfı olarak toplamasını engeller

    TEST_SENARYOLARI = [
        {"girdi": "", "beklenen": "", "isim": "Boş String Sınırı"},
        {"girdi": "A", "beklenen": "A1", "isim": "Tek Karakter"},
        {"girdi": "AABBC", "beklenen": "A2B2C1", "isim": "Tekrarlı Standart Dizi"},
        {"girdi": "ABCD", "beklenen": "A1B1C1D1", "isim": "Tekrarsız Dizi"},
    ]

    @classmethod
    def testleri_kostur(cls, kod_metni: str) -> Dict[str, Any]:
        """
        Verilen kod metnini yerel isim alanında derler ve testleri çalıştırır.
        """
        yerel_alan = {}
        try:
            exec(kod_metni, {}, yerel_alan)
        except Exception as e:
            return {
                "derleme_hatasi": True,
                "tum_testler_gecti_mi": False,
                "gecen_sayisi": 0,
                "toplam_test_sayisi": len(cls.TEST_SENARYOLARI),
                "hata_raporu": f"Sözdizimi/Derleme Hatası: {str(e)}",
                "ayrintilar": [],
            }

        fn = yerel_alan.get("run_length_encoding")
        if not fn:
            return {
                "derleme_hatasi": True,
                "tum_testler_gecti_mi": False,
                "gecen_sayisi": 0,
                "toplam_test_sayisi": len(cls.TEST_SENARYOLARI),
                "hata_raporu": "'run_length_encoding' fonksiyonu bulunamadı!",
                "ayrintilar": [],
            }

        gecenler = 0
        ayrintilar = []
        ilk_hata_mesaji = None

        for test in cls.TEST_SENARYOLARI:
            isim = test["isim"]
            girdi = test["girdi"]
            beklenen = test["beklenen"]

            try:
                cikti = fn(girdi)
                if cikti == beklenen:
                    gecenler += 1
                    ayrintilar.append({"isim": isim, "durum": "PASSED", "hata": None})
                else:
                    hata_metni = f"AssertionError: Girdi='{girdi}' için Çıktı='{cikti}', Beklenen='{beklenen}'"
                    ayrintilar.append({"isim": isim, "durum": "FAILED", "hata": hata_metni})
                    if not ilk_hata_mesaji:
                        ilk_hata_mesaji = hata_metni
            except Exception as e:
                hata_metni = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                ayrintilar.append({"isim": isim, "durum": "ERROR", "hata": hata_metni})
                if not ilk_hata_mesaji:
                    ilk_hata_mesaji = hata_metni

        return {
            "derleme_hatasi": False,
            "tum_testler_gecti_mi": (gecenler == len(cls.TEST_SENARYOLARI)),
            "gecen_sayisi": gecenler,
            "toplam_test_sayisi": len(cls.TEST_SENARYOLARI),
            "hata_raporu": ilk_hata_mesaji,
            "ayrintilar": ayrintilar,
        }
