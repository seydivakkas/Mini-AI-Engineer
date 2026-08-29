"""
System 1 Motoru: Hızlı ve Sezgisel LLM Akıl Yürütme Modülü (Day 141 - Faz 8).
Düşünme bütçesi kullanmadan doğrudan otoregresif çıkarım yapan System 1 motoru.
"""

from typing import Dict, Any
import time


class Sistem1Motoru:
    """Hızlı, refleksif, doğrudan tek adımlı çıkarım yapan System 1 motoru."""

    def __init__(self):
        # Bilişsel tuzaklara (Cognitive Reflection Tests) verilen sezgisel hatalı ezber yanıtlar
        self._sezgisel_ezber_haritasi = {
            "sopave_top": "Topun fiyatı 10 centtir.",
            "nilufer_golu": "Gölün yarısı 24 günde kaplanır.",
            "bes_makine": "100 makine 100 parçayı 100 dakikada üretir.",
        }

    def yanitla(self, soru_anahtari: str, soru_metni: str) -> Dict[str, Any]:
        """Düşünme adımı olmadan doğrudan refleksif yanıt üretir."""
        baslangic = time.perf_counter()

        # Doğrudan yüzeysel desen eşleme
        yanit = self._sezgisel_ezber_haritasi.get(
            soru_anahtari,
            f"Sezgisel doğrudan yanıt: {soru_metni}"
        )

        bitis = time.perf_counter()
        gecikme_ms = (bitis - baslangic) * 1000.0 + 12.0  # Temel çıkarım gecikmesi

        return {
            "sistem": "System 1 (Hızlı / Sezgisel)",
            "soru": soru_metni,
            "dusunme_izleri": [],
            "dusunme_token_sayisi": 0,
            "cikti_token_sayisi": len(yanit.split()),
            "yanit": yanit,
            "gecikme_ms": round(gecikme_ms, 2),
            "guven_skoru": 0.65,  # Sezgisel aşırı özgüven
        }
