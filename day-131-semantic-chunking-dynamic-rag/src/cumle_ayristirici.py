"""
Cümle Ayrıştırıcı ve Bağlam Tamponlayıcı Modülü (Day 131 - Faz 7).
Ham metinleri dilbilgisel ve noktalama kurallarına göre cümlelere bölen ve yerel bağlam tamponu oluşturan modül.
"""

import re
from typing import List, Dict, Any


class CumleAyristirici:
    """Metinleri sağlam regex kurallarıyla temiz cümle dizilerine ayrıştırır."""

    # Nokta, ünlem, soru işareti ve satır sonu ayracı (Kısaltmaları korumaya yönelik filtre)
    CUMLE_AYRAC_DESENI = re.compile(r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9"\'\(\[])')

    @classmethod
    def ayristir(cls, metin: str) -> List[str]:
        """Ham metni temizlenmiş cümleler listesine böler."""
        if not metin or not metin.strip():
            return []

        # Satır başı/sonu boşlukları temizle ve çoklu boşlukları tekle
        temiz_metin = re.sub(r'\s+', ' ', metin.strip())

        # Cümle sınırlarından böl
        ham_cumleler = cls.CUMLE_AYRAC_DESENI.split(temiz_metin)

        # Boşlukları temizle ve çok kısa anlamsız parçaları ele
        gecerli_cumleler = [c.strip() for c in ham_cumleler if len(c.strip()) > 3]
        return gecerli_cumleler


class BaglamTamponlayici:
    """Tekil cümlelerin bağlamını zenginleştirmek için çevre cümlelerle birleştiren tamponlayıcı."""

    @staticmethod
    def tampon_olustur(cumleler: List[str], tampon_boyutu: int = 1) -> List[Dict[str, Any]]:
        """
        Her cümlenin önüne ve arkasına 'tampon_boyutu' kadar komşu cümle ekleyerek
        zenginleştirilmiş bağlam metni üretir.
        """
        n = len(cumleler)
        zengin_listesi: List[Dict[str, Any]] = []

        for i in range(n):
            baslangic = max(0, i - tampon_boyutu)
            bitis = min(n, i + tampon_boyutu + 1)
            birlestirilmis_baglam = " ".join(cumleler[baslangic:bitis])

            zengin_listesi.append({
                "cumle_indeksi": i,
                "ana_cumle": cumleler[i],
                "birlestirilmis_baglam": birlestirilmis_baglam,
            })

        return zengin_listesi
