"""
ReAct Scratchpad Bellek Modülü (Day 121 - Faz 7).
Ajanın düşünce, eylem ve gözlem geçmişini (Trajectory) saklayan ve bağlam optimizasyonu yapan bellek tamponu.
"""

from typing import List, Dict, Any


class AdimKaydi:
    """Tek bir ReAct döngü adımının verilerini tutar."""

    def __init__(
        self,
        adim_no: int,
        dusunce: str,
        arac_adi: str = None,
        arac_girdisi: str = None,
        gozlem: str = None,
        nihai_yanit: str = None,
    ):
        self.adim_no = adim_no
        self.dusunce = dusunce
        self.arac_adi = arac_adi
        self.arac_girdisi = arac_girdisi
        self.gozlem = gozlem
        self.nihai_yanit = nihai_yanit

    def metne_dok(self) -> str:
        parcalar = []
        if self.dusunce:
            parcalar.append(f"Thought {self.adim_no}: {self.dusunce}")
        if self.arac_adi and self.arac_girdisi:
            parcalar.append(f"Action {self.adim_no}: {self.arac_adi}[{self.arac_girdisi}]")
        if self.gozlem:
            parcalar.append(f"Observation {self.adim_no}: {self.gozlem}")
        if self.nihai_yanit:
            parcalar.append(f"Final Answer: {self.nihai_yanit}")
        return "\n".join(parcalar)


class ScratchpadBellek:
    """Ajanın çalışma alanı (Scratchpad) belleğini yönetir."""

    def __init__(self, maksimum_adim: int = 10):
        self.maksimum_adim = maksimum_adim
        self.gecmis: List[AdimKaydi] = []

    def adim_ekle(self, adim: AdimKaydi):
        self.gecmis.append(adim)
        if len(self.gecmis) > self.maksimum_adim:
            self.gecmis.pop(0)  # Kaydırma penceresi (Sliding Window)

    def sifirla(self):
        self.gecmis.clear()

    def metin_olarak_al(self) -> str:
        """Tüm scratchpad geçmişini tek bir biçimlendirilmiş metin olarak döndürür."""
        if not self.gecmis:
            return ""
        return "\n\n".join([adim.metne_dok() for adim in self.gecmis])

    def toplam_adim_sayisi(self) -> int:
        return len(self.gecmis)

    def son_adim(self) -> AdimKaydi:
        return self.gecmis[-1] if self.gecmis else None
