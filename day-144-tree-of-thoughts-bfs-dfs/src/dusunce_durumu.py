"""
Düşünce Durumu (Thought State) Modülü (Day 144 - Faz 8).
Tree of Thoughts (ToT) ağaç düğüm veri yapısı ve durum yönetimi.
"""

from typing import List, Optional, Dict, Any


class DusunceDurumu:
    """Arama ağacındaki her bir ara düşünce durumunu temsil eden düğüm."""

    def __init__(
        self,
        durum_id: str,
        sayilar: List[float],
        adim_gecmisi: List[str] = None,
        ebeveyn_id: Optional[str] = None,
        derinlik: int = 0,
    ):
        self.durum_id = durum_id
        self.sayilar = [float(s) for s in sayilar]
        self.adim_gecmisi = adim_gecmisi or []
        self.ebeveyn_id = ebeveyn_id
        self.derinlik = derinlik

        self.deger_puani: float = 0.0
        self.degerlendirme: str = "degerlendirilmedi"  # "kesin", "olası", "imkansiz"
        self.cocuk_durumlar: List["DusunceDurumu"] = []

    def hedefe_ulasti_mi(self, hedef: float = 24.0, tolerans: float = 1e-4) -> bool:
        """Kalan tek bir sayı varsa ve hedefe eşitse çözüm bulunmuştur."""
        if len(self.sayilar) == 1:
            return abs(self.sayilar[0] - hedef) < tolerans
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serileştirilmiş sözlük çıktısı."""
        return {
            "durum_id": self.durum_id,
            "sayilar": self.sayilar,
            "adim_gecmisi": self.adim_gecmisi,
            "ebeveyn_id": self.ebeveyn_id,
            "derinlik": self.derinlik,
            "deger_puani": round(self.deger_puani, 3),
            "degerlendirme": self.degerlendirme,
        }
