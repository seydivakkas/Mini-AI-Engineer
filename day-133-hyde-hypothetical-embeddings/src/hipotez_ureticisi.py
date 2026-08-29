"""
HyDE Hipotez Belgesi Üreticisi (Day 133 - Faz 7).
Kullanıcı sorusundan sıfır-atış (zero-shot) varsayımsal yanıt pasajları üreten motor.
"""

from typing import List, Dict
import re


class HipotezUreticisi:
    """Kullanıcı sorusunu belge uzayına projekte etmek için varsayımsal pasajlar üretir."""

    PERSPEKTIFLER = [
        "tanimsal_mekanik",
        "algoritmik_uygulama",
        "mimari_optimizasyon",
    ]

    @classmethod
    def tek_hipotez_uret(cls, sorgu: str, perspektif: str = "tanimsal_mekanik") -> str:
        """Sorgu için belirli bir teknik perspektifte varsayımsal belge pasajı üretir."""
        temiz_sorgu = re.sub(r"[^\w\s]", "", sorgu).strip()
        anahtar_kelimeler = [w for w in temiz_sorgu.split() if len(w) > 2]
        konu = " ".join(anahtar_kelimeler[:4]) if anahtar_kelimeler else "konu"

        if perspektif == "tanimsal_mekanik":
            return (
                f"{konu.capitalize()} teknolojisi, modern dağıtık ve derin öğrenme sistemlerinde "
                f"temel bir bileşendir. Matematiksel olarak yüksek boyutlu vektör temsilleri ve "
                f"özel katsayı matrisleri üzerinden hesaplama gerçekleştirir. "
                f"Temel çalışma prensibi veri tutarlılığı ve düşük gecikmeli işlem kapasitesidir."
            )
        elif perspektif == "algoritmik_uygulama":
            return (
                f"Uygulama düzeyinde {konu} mekanizması; veri yapıları üzerinde optimize edilmiş "
                f"arama algoritmaları, önbellekleme katmanları ve tensör işlemlerini koordine eder. "
                f"Geleneksel yöntemlere kıyasla işlem karmaşıklığını O(N)'den O(log N)'e indirir."
            )
        else:  # mimari_optimizasyon
            return (
                f"Sistem mimarisinde {konu} bileşeni; GPU/CPU bellek bant genişliğini maksimize ederken "
                f"donanım hızlandırıcıları ve paralel iş parçacıkları ile iletişim yükünü minimize eder. "
                f"Yüksek verimlilik, kuantizasyon ve paralel hesaplama desteği sunar."
            )

    @classmethod
    def coklu_hipotez_uret(cls, sorgu: str, n: int = 3) -> List[str]:
        """Sorgu için n adet farklı teknik perspektiften varsayımsal belge üretir."""
        hipotezler = []
        for i in range(n):
            perspektif = cls.PERSPEKTIFLER[i % len(cls.PERSPEKTIFLER)]
            hipotezler.append(cls.tek_hipotez_uret(sorgu, perspektif=perspektif))
        return hipotezler
