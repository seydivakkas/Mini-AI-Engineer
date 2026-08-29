"""
Hiyerarşik Parçalayıcı (Parent-Child Chunking) Modülü (Day 132 - Faz 7).
Metinleri büyük ebeveyn parçalara (Parent) ve bunların altındaki küçük çocuk parçalara (Child) bölen yapı.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import re


@dataclass
class CocukParca:
    """Vektör araması için kullanılan küçük ve keskin çocuk parça."""
    child_id: str
    parent_id: str
    metin: str
    karakter_sayisi: int


@dataclass
class EbeveynParca:
    """LLM yanıt üretimi için bağlam sağlayan zengin ebeveyn parça."""
    parent_id: str
    metin: str
    karakter_sayisi: int
    cocuk_idleri: List[str] = field(default_factory=list)


class HiyerarsikParcalayici:
    """Belgeyi ebeveyn ve çocuk parçalardan oluşan 2 katmanlı hiyerarşik ağaca böler."""

    def __init__(
        self,
        ebeveyn_boyutu: int = 600,
        ebeveyn_cakisim: int = 80,
        cocuk_boyutu: int = 160,
        cocuk_cakisim: int = 30,
    ):
        self.ebeveyn_boyutu = ebeveyn_boyutu
        self.ebeveyn_cakisim = ebeveyn_cakisim
        self.cocuk_boyutu = cocuk_boyutu
        self.cocuk_cakisim = cocuk_cakisim

    def hiyerarsi_olustur(self, ham_metin: str) -> Tuple[List[EbeveynParca], List[CocukParca]]:
        """Ham metinden ebeveyn ve çocuk parçaları hiyerarşik olarak türetir."""
        ebeveynler: List[EbeveynParca] = []
        cocuklar: List[CocukParca] = []

        # 1. Ebeveyn Parçaları Üret
        baslangic = 0
        metin_len = len(ham_metin)
        parent_idx = 1

        while baslangic < metin_len:
            bitis = min(metin_len, baslangic + self.ebeveyn_boyutu)
            p_metin = ham_metin[baslangic:bitis].strip()
            p_id = f"PARENT_{parent_idx:03d}"

            ebeveyn = EbeveynParca(
                parent_id=p_id,
                metin=p_metin,
                karakter_sayisi=len(p_metin),
            )

            # 2. Bu Ebeveyn İçindeki Çocuk Parçaları Üret
            c_baslangic = 0
            p_len = len(p_metin)
            child_sub_idx = 1

            while c_baslangic < p_len:
                c_bitis = min(p_len, c_baslangic + self.cocuk_boyutu)
                c_metin = p_metin[c_baslangic:c_bitis].strip()
                c_id = f"{p_id}_CHILD_{child_sub_idx:02d}"

                if len(c_metin) > 10:  # Çok kısa artık parçaları yoksay
                    cocuk = CocukParca(
                        child_id=c_id,
                        parent_id=p_id,
                        metin=c_metin,
                        karakter_sayisi=len(c_metin),
                    )
                    cocuklar.append(cocuk)
                    ebeveyn.cocuk_idleri.append(c_id)
                    child_sub_idx += 1

                c_baslangic += self.cocuk_boyutu - self.cocuk_cakisim

            ebeveynler.append(ebeveyn)
            parent_idx += 1
            baslangic += self.ebeveyn_boyutu - self.ebeveyn_cakisim

        return ebeveynler, cocuklar
