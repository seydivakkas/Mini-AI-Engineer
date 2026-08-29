"""
Durum Değerlendirici (Value / Evaluation Function) Modülü (Day 144 - Faz 8).
Bir düşünce durumunu hedefe ulaşabilirlik açısından değerlendiren ve budayan motor.
"""

from typing import Tuple, List
from .dusunce_durumu import DusunceDurumu


class DurumDegerlendirici:
    """Tree of Thoughts için sezgisel değer fonksiyonu (Value Function V(s))."""

    @classmethod
    def degerlendir(cls, durum: DusunceDurumu, hedef: float = 24.0) -> Tuple[float, str]:
        """
        Durumu inceler ve (deger_puani, etiket) döner.
        Etiketler: "kesin_cozum", "olasi", "imkansiz"
        """
        sayilar = durum.sayilar

        # 1. Kalan tek sayı hedefe eşit mi?
        if len(sayilar) == 1:
            if abs(sayilar[0] - hedef) < 1e-4:
                return 1.0, "kesin_cozum"
            else:
                return 0.0, "imkansiz"

        # 2. İki sayı kaldıysa 24 elde edilebilir mi?
        if len(sayilar) == 2:
            a, b = sayilar[0], sayilar[1]
            adaylar = [a + b, a - b, b - a, a * b]
            if abs(b) > 1e-5:
                adaylar.append(a / b)
            if abs(a) > 1e-5:
                adaylar.append(b / a)

            if any(abs(val - hedef) < 1e-4 for val in adaylar):
                return 0.95, "olasi"

        # 3. Genel sezgisel kontrol (24'ün çarpanları: 6*4, 8*3, 12*2 vb.)
        carpanlar = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0]
        if any(any(abs(s - c) < 1e-4 for c in carpanlar) for s in sayilar):
            return 0.75, "olasi"

        # 4. Aşırı büyük veya imkansız negatif durumlar
        if all(s > 100 for s in sayilar) or any(s < 0 for s in sayilar):
            return 0.10, "imkansiz"

        return 0.50, "olasi"
