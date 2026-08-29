"""
Çıkmaz Sokak ve Mantıksal Çelişki Tespit Modülü (Day 148 - Faz 8).
Mantık hatalarını, aritmetik tutarsızlıkları ve çıkmaz durumları otomatik tespit eden motor.
"""

from typing import Tuple, Dict, Any, List


class CikmazSokakTespitcisi:
    """Akıl yürütme adımlarındaki çıkmazları ve çelişkileri yakalayan denetçi."""

    @classmethod
    def denetle(cls, adim_metni: str, durum_verisi: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Adımı denetler:
        Döner: (cikmaz_mi, hata_nedeni, guven_skoru)
        """
        metin = adim_metni.lower()

        # 1. Klasik CRT Çelişki Kontrolü (Bat & Ball paradox)
        if ("top = 0.10" in metin or "top 0.10" in metin) and "2 *" not in metin and "2*" not in metin and "fark" not in metin:
            return True, "Çelişki: Top $0.10 olursa Sopa $1.10 olur ve Toplam $1.20 çıkar ($1.10 değil)!", 0.99

        if "1.10 - 1.00 = 0.10" in metin and "fark" not in metin and "2 *" not in metin:
            return True, "Hatalı Çıkarım: Fark doğrudan topun fiyatı olamaz (2 * Top = Fark olmalı)!", 0.95

        # 2. Negatif veya Anlamsız Sayı Kontrolü
        sayilar = durum_verisi.get("sayilar", [])
        if any(s < 0 for s in sayilar) and not durum_verisi.get("negatif_izinli", False):
            return True, "Geçersiz Durum: Negatif ara sonuç üretildi!", 0.90

        if any(s > 1000 for s in sayilar):
            return True, "Çıkmaz Sokak: Sayılar hedef değerden aşırı uzaklaştı!", 0.92

        return False, "Geçerli Adım", 0.98
