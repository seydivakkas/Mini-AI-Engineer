"""
Ses Diyalog ve Gecikme Değerlendirici Modülü (Day 171 - FAZ 9).
End-to-End Latency, RTF (Real-Time Factor) ve Akustik Kalite Skorlarını hesaplar.
"""

from typing import Dict, Any


class SesDiyalogDegerlendirici:
    """Speech-to-Speech LLM Performans ve Gecikme Motoru."""

    @classmethod
    def rtf_hesapla(cls, uretim_suresi_ms: float, ses_uzunlugu_ms: float) -> float:
        """
        Real-Time Factor (RTF):
        RTF = Üretim Süresi / Ses Süresi (< 1.0 ise gerçek zamanlıdan hızlı)
        """
        if ses_uzunlugu_ms <= 0:
            return 0.0
        return round(float(uretim_suresi_ms / ses_uzunlugu_ms), 3)

    @classmethod
    def gecikme_tasarruf_orani(cls, geleneksel_ms: float, duallm_ms: float) -> float:
        """Geleneksel boru hattına göre kazanılan hız faktörü."""
        if duallm_ms <= 0:
            return 1.0
        return round(float(geleneksel_ms / duallm_ms), 1)
