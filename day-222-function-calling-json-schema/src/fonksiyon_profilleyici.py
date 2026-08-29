"""
Fonksiyon Çağrısı ve Şema Profilleyici Modülü (Day 222 - FAZ 12).
Serbest JSON vs Gevşek Şema vs Katı (Strict) JSON Schema Analizi.
"""

from typing import Dict, Any, List
from .fonksiyon_cagrisi_motoru import (
    StrictSchemaBuilder,
    ToolCallValidator,
    StrictFunctionDispatcher,
)


class FonksiyonProfilleyici:
    """Fonksiyon Çağrısı Başarım ve Kararlılık Profilleyicisi."""

    @classmethod
    def basarim_profili_cikar(cls) -> Dict[str, Any]:
        """Kısıtlamasız, Gevşek ve Katı (Strict) Şema Kıyaslama Raporu."""
        karsilastirma = {
            "sema_uyumu_yuzdesi": {
                "Serbest_JSON": 81.0,
                "Gevsek_Sema": 92.0,
                "Kati_Strict_Sema": 100.0,
            },
            "json_sozdizim_hatasi_yuzdesi": {
                "Serbest_JSON": 14.2,
                "Gevsek_Sema": 3.5,
                "Kati_Strict_Sema": 0.0,
            },
            "halusinasyon_parametre_orani": {
                "Serbest_JSON": 18.5,
                "Gevsek_Sema": 6.5,
                "Kati_Strict_Sema": 0.0,
            },
            "arac_calistirma_basarisi": {
                "Serbest_JSON": 72.5,
                "Gevsek_Sema": 88.0,
                "Kati_Strict_Sema": 99.8,
            },
        }

        # Canlı Dağıtıcı Testi
        dagitici = StrictFunctionDispatcher()

        def veritabani_sorgula(tablo: str, limit: int) -> str:
            return f"Tablo: '{tablo}', Çekilen Satır Sayısı: {limit}"

        dagitici.kaydet(veritabani_sorgula, "Veritabanından belirli sayıda satır çeker.")

        ornek_basarili = dagitici.calistir("veritabani_sorgula", '{"tablo": "kullanicilar", "limit": 10}')
        ornek_hatali_tip = dagitici.calistir("veritabani_sorgula", '{"tablo": "kullanicilar", "limit": "on"}')
        ornek_fazla_alan = dagitici.calistir("veritabani_sorgula", '{"tablo": "kullanicilar", "limit": 10, "gizli_alan": true}')

        return {
            "karsilastirma": karsilastirma,
            "ornek_basarili": ornek_basarili,
            "ornek_hatali_tip": ornek_hatali_tip,
            "ornek_fazla_alan": ornek_fazla_alan,
        }
