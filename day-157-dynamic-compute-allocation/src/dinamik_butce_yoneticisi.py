"""
Dinamik Token Bütçesi ve Hesaplama Rotalayıcı Modülü (Day 157 - Faz 8).
Zorluk skoruna göre dinamik token bütçesini, düşünce modunu ve tahmini çıkarım maliyetini belirler.
"""

from typing import Dict, Any, List
from .zorluk_tahmincisi import ZorlukTahmincisi


class DinamikButceYoneticisi:
    """Soru zorluğuna göre optimal test-time compute bütçesi tahsis eden rotalayıcı."""

    PROFILLER = {
        "Kolay": {
            "cikarim_modu": "Doğrudan Yanıt (System 1 Direct)",
            "token_butcesi": 32,
            "arama_derinligi": 0,
            "tahmini_gecikme_ms": 40.0,
            "tahmini_maliyet_tl": 0.002,
        },
        "Orta": {
            "cikarim_modu": "Standart CoT (Chain-of-Thought)",
            "token_butcesi": 512,
            "arama_derinligi": 1,
            "tahmini_gecikme_ms": 320.0,
            "tahmini_maliyet_tl": 0.025,
        },
        "Zor": {
            "cikarim_modu": "Derin Akıl Yürütme (MCTS & Tree Search)",
            "token_butcesi": 4096,
            "arama_derinligi": 8,
            "tahmini_gecikme_ms": 2400.0,
            "tahmini_maliyet_tl": 0.200,
        },
    }

    @classmethod
    def butce_tahsis_et(cls, soru_metni: str) -> Dict[str, Any]:
        """
        Soru için zorluğu tahmin eder ve dinamik compute bütçesi çıkarır.
        """
        zorluk_bilgisi = ZorlukTahmincisi.zorluk_hesapla(soru_metni)
        kategori = zorluk_bilgisi["kategori"]
        profil = cls.PROFILLER[kategori]

        return {
            "soru": soru_metni,
            "zorluk_skoru": zorluk_bilgisi["zorluk_skoru"],
            "kategori": kategori,
            "cikarim_modu": profil["cikarim_modu"],
            "tahsis_edilen_token_butcesi": profil["token_butcesi"],
            "arama_derinligi": profil["arama_derinligi"],
            "tahmini_gecikme_ms": profil["tahmini_gecikme_ms"],
            "tahmini_maliyet_tl": profil["tahmini_maliyet_tl"],
            "aciklama": zorluk_bilgisi["aciklama"],
        }
