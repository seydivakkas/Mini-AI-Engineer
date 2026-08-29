"""
Çıkarım Simülasyonu ve Tasarruf Kıyaslama Modülü (Day 157 - Faz 8).
Sabit bütçe ile dinamik bütçe yaklaşımlarını maliyet, gecikme ve token tüketimi açısından kıyaslar.
"""

from typing import Dict, Any, List
from .dinamik_butce_yoneticisi import DinamikButceYoneticisi


class CikarimSimulasyonu:
    """Sabit vs Dinamik Compute tasarruf simülatörü."""

    @classmethod
    def calistir(cls, sorular: List[str]) -> Dict[str, Any]:
        """
        Soru listesi üzerinde dinamik bütçe tahsisi yapar ve sabit bütçeyle kıyaslar.
        """
        sonuclar = []

        toplam_dinamik_token = 0
        toplam_dinamik_sure_ms = 0.0
        toplam_dinamik_maliyet_tl = 0.0

        sabit_token_per_soru = 4096
        sabit_sure_ms_per_soru = 2400.0
        sabit_maliyet_tl_per_soru = 0.200

        for s in sorular:
            tahsis = DinamikButceYoneticisi.butce_tahsis_et(s)
            sonuclar.append(tahsis)

            toplam_dinamik_token += tahsis["tahsis_edilen_token_butcesi"]
            toplam_dinamik_sure_ms += tahsis["tahmini_gecikme_ms"]
            toplam_dinamik_maliyet_tl += tahsis["tahmini_maliyet_tl"]

        toplam_soru = len(sorular)
        toplam_sabit_token = toplam_soru * sabit_token_per_soru
        toplam_sabit_sure_ms = toplam_soru * sabit_sure_ms_per_soru
        toplam_sabit_maliyet_tl = toplam_soru * sabit_maliyet_tl_per_soru

        token_tasarrufu_yuzde = ((toplam_sabit_token - toplam_dinamik_token) / toplam_sabit_token) * 100.0
        maliyet_tasarrufu_yuzde = ((toplam_sabit_maliyet_tl - toplam_dinamik_maliyet_tl) / toplam_sabit_maliyet_tl) * 100.0
        hizlanma_orani = toplam_sabit_sure_ms / max(1.0, toplam_dinamik_sure_ms)

        return {
            "soru_sonuclari": sonuclar,
            "toplam_soru_sayisi": toplam_soru,
            "toplam_dinamik_token": toplam_dinamik_token,
            "toplam_sabit_token": toplam_sabit_token,
            "toplam_dinamik_maliyet_tl": toplam_dinamik_maliyet_tl,
            "toplam_sabit_maliyet_tl": toplam_sabit_maliyet_tl,
            "toplam_dinamik_sure_ms": toplam_dinamik_sure_ms,
            "toplam_sabit_sure_ms": toplam_sabit_sure_ms,
            "token_tasarrufu_yuzde": round(token_tasarrufu_yuzde, 1),
            "maliyet_tasarrufu_yuzde": round(maliyet_tasarrufu_yuzde, 1),
            "hizlanma_orani": round(hizlanma_orani, 2),
        }
