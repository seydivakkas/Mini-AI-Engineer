"""
İlişki Çıkarıcı (Relationship Triplet Extraction) Modülü (Day 136 - Faz 7 - GraphRAG-1).
Varlıklar arasındaki Özne-Yüklem-Nesne (Subject-Predicate-Object) üçlülerini çıkaran modül.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re

from .varlik_cikarici import Varlik


@dataclass
class IliskiUclusu:
    """Bilgi Grafı Kenarı (Edge / Triplet: (Subject, Predicate, Object))."""
    ozne: str
    yuklem: str
    nesne: str
    agirlik: float = 1.0
    baglam: str = ""


class IliskiCikarici:
    """Metindeki cümle bağlamlarını inceleyerek varlıklar arası anlamsal ilişkileri kurar."""

    YUKLEM_KALIPLARI: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(kullanır|kullanarak|yararlanır)\b", re.I), "KULLANIR"),
        (re.compile(r"\b(engeller|önler|durdurur)\b", re.I), "ENGELLER"),
        (re.compile(r"\b(uygular|çalıştırır|icra eder)\b", re.I), "UYGULAR"),
        (re.compile(r"\b(hızlandırır|iyileştirir|artırır)\b", re.I), "HIZLANDIRIR"),
        (re.compile(r"\b(destekler|barındırır|içerir)\b", re.I), "DESTEKLER"),
        (re.compile(r"\b(bağlıdır|ilişkilidir|yönetir)\b", re.I), "YONETIR"),
    ]

    @classmethod
    def cikar(cls, metin: str, varliklar: List[Varlik]) -> List[IliskiUclusu]:
        """Cümle bazında eş-zamanlı geçen varlık çiftleri arasındaki ilişkileri tespit eder."""
        cumleler = re.split(r"(?<=[.!?])\s+", metin)
        varlik_isimleri = {v.isim: v for v in varliklar}

        ucluler: Dict[Tuple[str, str, str], IliskiUclusu] = {}

        for cumle in cumleler:
            # Bu cümlede geçen varlıkları tespit et
            gecen_varliklar = []
            for v_isim, v_obj in varlik_isimleri.items():
                olasi_adlar = [v_isim] + v_obj.aliaslar
                for ad in olasi_adlar:
                    if re.search(r"\b" + re.escape(ad) + r"\b", cumle, re.IGNORECASE):
                        gecen_varliklar.append(v_isim)
                        break

            # Cümlede en az 2 varlık varsa aralarındaki yüklemi bul
            if len(gecen_varliklar) >= 2:
                yuklem = "ILISKILIDIR"  # Varsayılan
                for regex, yuklem_adi in cls.YUKLEM_KALIPLARI:
                    if regex.search(cumle):
                        yuklem = yuklem_adi
                        break

                for i in range(len(gecen_varliklar)):
                    for j in range(i + 1, len(gecen_varliklar)):
                        ozne = gecen_varliklar[i]
                        nesne = gecen_varliklar[j]
                        anahtar = (ozne, yuklem, nesne)

                        if anahtar not in ucluler:
                            ucluler[anahtar] = IliskiUclusu(
                                ozne=ozne,
                                yuklem=yuklem,
                                nesne=nesne,
                                agirlik=1.0,
                                baglam=cumle.strip(),
                            )
                        else:
                            ucluler[anahtar].agirlik += 0.5

        return list(ucluler.values())
