"""
Soru Zorluğu ve Karmaşıklık Tahmin Edici Modülü (Day 157 - Faz 8).
Girdi sorgusunun semantik yapısını, kısıt sayısını ve bilişsel yükünü analiz ederek zorluk skoru (0.0 - 1.0) üretir.
"""

from typing import Dict, Any, List


class ZorlukTahmincisi:
    """Sorgunun zorluk seviyesini ve kategorisini belirleyen motor."""

    ZOR_ANAHTARLAR = ["ispat", "algoritma", "optimize", "teorem", "polinom", "dinamik programlama", "aime", "karmaşıklık"]
    ORTA_ANAHTARLAR = ["hesapla", "kaçtır", "adım", "yüzde", "oran", "çarpım", "kdv", "faiz", "karşılaştır"]

    @classmethod
    def zorluk_hesapla(cls, soru_metni: str) -> Dict[str, Any]:
        """
        Soru metnini analiz ederek zorluk skorunu ve kategorisini döner.
        """
        metin = soru_metni.lower()
        kelime_sayisi = len(metin.split())

        # 1. Zor Sorgu Taraması
        zor_eslesme = sum(1 for k in cls.ZOR_ANAHTARLAR if k in metin)
        if zor_eslesme >= 1 or kelime_sayisi > 40:
            skor = min(1.0, 0.75 + (0.05 * zor_eslesme))
            kategori = "Zor"
            aciklama = "Derin akıl yürütme, çoklu adımlar veya matematiksel/algoritmik ispat gerektirir."

        # 2. Orta Sorgu Taraması
        elif any(k in metin for k in cls.ORTA_ANAHTARLAR) or kelime_sayisi >= 15:
            skor = 0.45
            kategori = "Orta"
            aciklama = "Standart düşünce zinciri (Chain-of-Thought) ve 2-4 adımlı aritmetik gerektirir."

        # 3. Kolay Sorgu (Doğrudan Bilgi / Factual Recall)
        else:
            skor = 0.15
            kategori = "Kolay"
            aciklama = "Doğrudan parametrik hafıza sorgusu; ek düşünme tokenı gerektirmez."

        return {
            "soru": soru_metni,
            "zorluk_skoru": skor,
            "kategori": kategori,
            "kelime_sayisi": kelime_sayisi,
            "aciklama": aciklama,
        }
