"""
LangGraph Durum Şeması ve İndirgeyici (State Reducer) Modülü (Day 127 - Faz 7).
Tip güvenli durum sözlüğü, mesaj kanalları ve düğüm geçişlerinde durum birleştirme mantığı.
"""

from typing import Dict, Any, List, Optional
import copy


class DurumIndirgeyici:
    """Düğüm çıktılarındaki güncellemeleri mevcut çizge durumuna uygulayan indirgeyici."""

    @staticmethod
    def indirge(eski_durum: Dict[str, Any], guncellemeler: Dict[str, Any]) -> Dict[str, Any]:
        """
        Durum alanlarını güvenle birleştirir.
        Listeler (mesajlar, adim_gecmisi) uç uca eklenir, skaler değerler güncellenir.
        """
        yeni_durum = copy.deepcopy(eski_durum)

        for anahtar, deger in guncellemeler.items():
            if anahtar in ["mesajlar", "adim_gecmisi"] and isinstance(deger, list):
                if anahtar not in yeni_durum or not isinstance(yeni_durum[anahtar], list):
                    yeni_durum[anahtar] = []
                yeni_durum[anahtar].extend(deger)
            else:
                yeni_durum[anahtar] = deger

        return yeni_durum


def varsayilan_durum_olustur(
    musteri_id: str = "MUST_101",
    talep_tutari: float = 0.0,
    talep_turu: str = "iade",
    baslangic_mesaji: str = "",
) -> Dict[str, Any]:
    """İş akışı için sıfır başlangıç durumunu üretir."""
    return {
        "musteri_id": musteri_id,
        "talep_tutari": talep_tutari,
        "talep_turu": talep_turu,
        "risk_skoru": 0.0,
        "onay_gerekli_mi": False,
        "insan_onayladi_mi": None,
        "odeme_yapildi_mi": False,
        "mesajlar": [{"rol": "kullanici", "icerik": baslangic_mesaji}] if baslangic_mesaji else [],
        "adim_gecmisi": ["START"],
        "nihai_durum": "ISLENIYOR",
        "hata_mesaji": None,
        "tekrar_sayisi": 0,
    }
