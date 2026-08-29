"""
Sadakat ve Halüsinasyon Ölçücü (Faithfulness / Groundedness) Modülü (Day 140 - Faz 7).
Ragas & TruLens mimarisinde üretilen yanıtın bağlama sadakatini (Groundedness) ölçen motor.
"""

from typing import List, Dict, Any, Tuple
import re


class SadakatOlcucu:
    """Üretilen yanıttaki iddiaları bağlamdaki kanıtlarla eşleyip sadakat skorunu hesaplar."""

    @classmethod
    def iddialara_ayir(cls, yanit: str) -> List[str]:
        """Yanıtı atomik iddia cümlelerine böler."""
        cumleler = re.split(r"(?<=[.!?])\s+", yanit.strip())
        return [c.strip() for c in cumleler if len(c.strip()) > 8]

    @classmethod
    def iddia_dogrula(cls, iddia: str, baglam_metinleri: List[str]) -> Tuple[bool, float]:
        """Bir iddianın bağlamda geçen kanıtlarla desteklenip desteklenmediğini ölçer."""
        iddia_kelimeleri = set(re.findall(r"\w+", iddia.lower()))
        if not iddia_kelimeleri:
            return False, 0.0

        tum_baglam = " ".join(baglam_metinleri).lower()
        baglam_kelimeleri = set(re.findall(r"\w+", tum_baglam))

        ortak = len(iddia_kelimeleri.intersection(baglam_kelimeleri))
        oran = ortak / max(1, len(iddia_kelimeleri))

        # En az %60 oranında bağlam eşleşmesi varsa iddia desteklenmiştir
        desteklendi = oran >= 0.60
        return desteklendi, round(oran, 3)

    @classmethod
    def olc(cls, yanit: str, baglam_metinleri: List[str]) -> Dict[str, Any]:
        """
        Faithfulness Formülü: Desteklenen İddia Sayısı / Toplam İddia Sayısı
        """
        iddialar = cls.iddialara_ayir(yanit)
        if not iddialar:
            return {"sadakat_skoru": 1.0, "halusinasyon_orani": 0.0, "toplam_iddia": 0}

        desteklenen_sayisi = 0
        desteklenen_iddialar = []
        halusinasyon_iddialar = []

        for iddia in iddialar:
            desteklendi, oran = cls.iddia_dogrula(iddia, baglam_metinleri)
            if desteklendi:
                desteklenen_sayisi += 1
                desteklenen_iddialar.append(iddia)
            else:
                halusinasyon_iddialar.append(iddia)

        sadakat = desteklenen_sayisi / len(iddialar)
        halusinasyon = 1.0 - sadakat

        return {
            "sadakat_skoru": round(sadakat, 4),
            "halusinasyon_orani": round(halusinasyon, 4),
            "toplam_iddia_sayisi": len(iddialar),
            "desteklenen_iddia_sayisi": desteklenen_sayisi,
            "desteklenen_iddialar": desteklenen_iddialar,
            "halusinasyon_iddialar": halusinasyon_iddialar,
        }
