"""
Varlık Çözümleyici ve Tekilleştirici Modülü (Day 136 - Faz 7 - GraphRAG-1).
Eşanlamlı, kısaltma ve varyasyon varlıkları tek bir kanonik düğümde (Canonical Node) birleştiren modül.
"""

from typing import List, Dict, Any, Tuple
import re

from .varlik_cikarici import Varlik
from .iliski_cikarici import IliskiUclusu


class VarlikCozumleyici:
    """Eşanlamlıları ve varyasyonları kanonik varlık adına eşler (Entity Resolution)."""

    KANONIK_ESLEME: Dict[str, str] = {
        "vit": "Vision Transformer",
        "vision transformers": "Vision Transformer",
        "postgres": "PostgreSQL",
        "lob": "Limit Order Book",
        "emir defteri": "Limit Order Book",
        "öz-dikkat": "Self-Attention",
        "öz dikkat": "Self-Attention",
        "raft protokolü": "Raft",
        "çoğunluk kuralı": "Quorum",
    }

    @classmethod
    def kanonik_ad(cls, ham_isim: str) -> str:
        """İsmi kanonik standart haline çevirir."""
        temiz = ham_isim.strip().lower()
        return cls.KANONIK_ESLEME.get(temiz, ham_isim)

    @classmethod
    def cozumle(
        cls, varliklar: List[Varlik], iliskiler: List[IliskiUclusu]
    ) -> Tuple[List[Varlik], List[IliskiUclusu], Dict[str, str]]:
        """
        Varlıkları tekilleştirir ve ilişkilerdeki özne/nesne isimlerini günceller.
        """
        esleme_haritasi: Dict[str, str] = {}
        tekil_varliklar: Dict[str, Varlik] = {}

        # 1. Varlıkları Kanonik Ada Göre Birleştir
        for v in varliklar:
            k_ad = cls.kanonik_ad(v.isim)
            esleme_haritasi[v.isim] = k_ad

            if k_ad not in tekil_varliklar:
                v.isim = k_ad
                tekil_varliklar[k_ad] = v
            else:
                tekil_varliklar[k_ad].frekans += v.frekans
                if v.aciklama and not tekil_varliklar[k_ad].aciklama:
                    tekil_varliklar[k_ad].aciklama = v.aciklama

        # 2. İlişkilerdeki İsimleri Güncelle
        guncel_iliskiler: Dict[Tuple[str, str, str], IliskiUclusu] = {}
        for iliski in iliskiler:
            yeni_ozne = esleme_haritasi.get(iliski.ozne, cls.kanonik_ad(iliski.ozne))
            yeni_nesne = esleme_haritasi.get(iliski.nesne, cls.kanonik_ad(iliski.nesne))

            if yeni_ozne != yeni_nesne:  # Kendine döngüyü engelle
                anahtar = (yeni_ozne, iliski.yuklem, yeni_nesne)
                if anahtar not in guncel_iliskiler:
                    guncel_iliskiler[anahtar] = IliskiUclusu(
                        ozne=yeni_ozne,
                        yuklem=iliski.yuklem,
                        nesne=yeni_nesne,
                        agirlik=iliski.agirlik,
                        baglam=iliski.baglam,
                    )
                else:
                    guncel_iliskiler[anahtar].agirlik += iliski.agirlik

        return list(tekil_varliklar.values()), list(guncel_iliskiler.values()), esleme_haritasi
