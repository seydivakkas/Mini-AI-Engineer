"""
Reflexion Episodik Hafıza Tamponu Modülü (Day 123 - Faz 7).
Geçmiş denemeleri (Trial), hataları ve sözel öz-eleştirileri saklayan ve prompta enjekte eden bellek tamponu.
"""

from typing import List, Dict, Any, Optional


class DenemeKaydi:
    """Tek bir Reflexion denemesinin (Trial) kayıt verisi."""

    def __init__(
        self,
        deneme_no: int,
        kod: str,
        odul: float,
        hata_mesaji: str,
        oz_elestiri: str,
    ):
        self.deneme_no = deneme_no
        self.kod = kod
        self.odul = odul
        self.hata_mesaji = hata_mesaji
        self.oz_elestiri = oz_elestiri


class ReflexionHafizaTamponu:
    """Ajanın önceki deneme geçmişini (Episodic Memory) yöneten tampon."""

    def __init__(self, maksimum_hafiza: int = 5):
        self.maksimum_hafiza = maksimum_hafiza
        self.denemeler: List[DenemeKaydi] = []

    def deneme_ekle(self, kayit: DenemeKaydi):
        self.denemeler.append(kayit)
        if len(self.denemeler) > self.maksimum_hafiza:
            self.denemeler.pop(0)

    def sifirla(self):
        self.denemeler.clear()

    def prompt_gecmisi_olustur(self) -> str:
        """Yeni deneme için geçmiş hataları ve çıkarılan dersleri biçimlendirir."""
        if not self.denemeler:
            return "Henüz önceki bir deneme ve hata kaydı bulunmamaktadır."

        satirlar = ["--- GEÇMİŞ DENEMELER VE ÇIKARILAN DERSLER (EPISODIC MEMORY) ---"]
        for d in self.denemeler:
            satirlar.append(f"\n[Deneme {d.deneme_no}]:")
            satirlar.append(f"• Alınan Ödül: {d.odul:.2f}/1.0")
            satirlar.append(f"• Hata Teşhisi: {d.hata_mesaji}")
            satirlar.append(f"• Çıkarılan Ders: {d.oz_elestiri}")
        satirlar.append("\nYukarıdaki dersleri dikkate alarak bu hataları TEKRAR ETMEYEN yeni bir kod üretin.")
        return "\n".join(satirlar)

    def en_iyi_deneme(self) -> Optional[DenemeKaydi]:
        if not self.denemeler:
            return None
        return max(self.denemeler, key=lambda x: x.odul)
