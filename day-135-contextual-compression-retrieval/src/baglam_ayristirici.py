"""
Bağlam Ayrıştırıcı ve Cümle Segmentasyon Modülü (Day 135 - Faz 7).
Getirilen belgeleri kaynak ID ve konum etiketleriyle cümle birimlerine ayıran modül.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import re


@dataclass
class CumleBirimi:
    """Tekil cümle ve kaynak metaverisi."""
    doc_id: str
    cumle_index: int
    metin: str
    karakter_sayisi: int
    token_tahmini: int


class BaglamAyristirici:
    """Belgeleri temiz cümle birimlerine ayrıştırır."""

    CUMLE_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])")

    @classmethod
    def ayristir(cls, doc_id: str, metin: str) -> List[CumleBirimi]:
        """Tek bir belgeyi cümle birimlerine böler."""
        temiz_metin = re.sub(r"\s+", " ", metin.strip())
        ham_cumleler = cls.CUMLE_REGEX.split(temiz_metin)

        cumle_birimleri: List[CumleBirimi] = []
        c_idx = 1

        for c in ham_cumleler:
            c_str = c.strip()
            if len(c_str) > 10:
                token_tahmini = max(1, len(c_str.split()) * 4 // 3)
                cumle_birimleri.append(
                    CumleBirimi(
                        doc_id=doc_id,
                        cumle_index=c_idx,
                        metin=c_str,
                        karakter_sayisi=len(c_str),
                        token_tahmini=token_tahmini,
                    )
                )
                c_idx += 1

        return cumle_birimleri

    @classmethod
    def toplu_ayristir(cls, belgeler: List[Dict[str, Any]]) -> List[CumleBirimi]:
        """Birden çok belgeyi ayrıştırıp düz liste halinde döndürür."""
        tum_cumleler = []
        for b in belgeler:
            tum_cumleler.extend(cls.ayristir(b["doc_id"], b["metin"]))
        return tum_cumleler
