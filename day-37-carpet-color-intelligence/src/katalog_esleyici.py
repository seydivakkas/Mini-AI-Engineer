"""
İplik Katalog Eşleyici ve Delta-E 2000 Tolerans Denetçisi (Yarn Catalog Matcher & QC).
"""

from typing import List, Dict, Any
from .delta_e_hesaplayici import delta_e_2000
from .renk_donusturucu import rgb_to_lab


class IplikKatalogEsleyici:
    """
    Çıkarılan halı iplik renklerini kurumsal iplik kartelası/kataloğu ile
    CIE Delta-E 2000 standardında eşler ve kalite tolerans raporu üretir.
    """

    # Standart Kurumsal İplik Kartelası (Referans Havuz)
    STANDART_KATALOG = [
        {"kod": "YARN-101", "ad": "Kraliyet Bordosu", "rgb": [138, 28, 48]},
        {"kod": "YARN-102", "ad": "Derin Gece Mavisi", "rgb": [24, 43, 73]},
        {"kod": "YARN-103", "ad": "Klasik Krem Vizon", "rgb": [228, 217, 198]},
        {"kod": "YARN-104", "ad": "Anadolu Hardal Sarısı", "rgb": [204, 154, 45]},
        {"kod": "YARN-105", "ad": "Osmanlı Zümrüt Yeşili", "rgb": [32, 98, 65]},
        {"kod": "YARN-106", "ad": "Antrasit Kömür Grisi", "rgb": [48, 52, 58]},
        {"kod": "YARN-107", "ad": "Fildişi Açık Bej", "rgb": [245, 240, 230]},
        {"kod": "YARN-108", "ad": "Kiremit Terracotta", "rgb": [184, 75, 41]}
    ]

    def __init__(self, ozel_katalog: List[Dict[str, Any]] = None, tolerans_esigi: float = 5.0):
        self.tolerans_esigi = tolerans_esigi
        katalog = ozel_katalog or self.STANDART_KATALOG

        self.katalog = []
        for urun in katalog:
            rgb = urun["rgb"]
            lab = rgb_to_lab([rgb])[0]
            self.katalog.append({
                "kod": urun["kod"],
                "ad": urun["ad"],
                "rgb": rgb,
                "lab": [float(round(c, 2)) for c in lab],
                "hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            })

    def esle_ve_raporla(self, cikarilan_iplikler: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Çıkarılan her iplik rengini en yakın katalog rengiyle eşler."""
        eslesmeler = []
        toplam_tolerans_ici = 0

        for iplik in cikarilan_iplikler:
            lab_iplik = iplik["lab"]

            en_iyi_eslesme = None
            en_kucuk_delta_e = float("inf")

            for kat in self.katalog:
                dE = delta_e_2000(lab_iplik, kat["lab"])
                if dE < en_kucuk_delta_e:
                    en_kucuk_delta_e = dE
                    en_iyi_eslesme = kat

            # Kalite Kararı
            if en_kucuk_delta_e < 2.0:
                kalite_durumu = "MUKEMMEL_UYUM"
                onay = True
            elif en_kucuk_delta_e < self.tolerans_esigi:
                kalite_durumu = "KABUL_EDILEBILIR"
                onay = True
            else:
                kalite_durumu = "PARTI_FARKI_RED"
                onay = False

            if onay:
                toplam_tolerans_ici += 1

            eslesmeler.append({
                "iplik_id": iplik["iplik_id"],
                "iplik_yuzdesi": iplik["yuzde"],
                "cikarilan_rgb": iplik["rgb"],
                "cikarilan_hex": iplik["hex"],
                "cikarilan_lab": iplik["lab"],
                "katalog_kod": en_iyi_eslesme["kod"],
                "katalog_ad": en_iyi_eslesme["ad"],
                "katalog_rgb": en_iyi_eslesme["rgb"],
                "katalog_hex": en_iyi_eslesme["hex"],
                "katalog_lab": en_iyi_eslesme["lab"],
                "delta_e_2000": float(round(en_kucuk_delta_e, 2)),
                "kalite_durumu": kalite_durumu,
                "onay": onay
            })

        genel_parti_onayi = (toplam_tolerans_ici == len(cikarilan_iplikler))

        return {
            "toplam_iplik_sayisi": len(cikarilan_iplikler),
            "tolerans_ici_iplik": toplam_tolerans_ici,
            "genel_parti_onayi": genel_parti_onayi,
            "eslesmeler": eslesmeler
        }
