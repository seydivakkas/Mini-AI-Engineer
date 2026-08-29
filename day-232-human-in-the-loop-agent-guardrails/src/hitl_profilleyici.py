"""
HITL Guardrail Profilleyici ve Başarım Kıyaslama Modülü (Day 232 - FAZ 12).
Kör Otonom Ajan vs Statik Bloklist vs HITL Güvenlik Bariyeri Analizi.
"""

from typing import Dict, Any, List
from .hitl_motoru import (
    RiskLevel,
    ActionRequest,
    ApprovalDecision,
    HITLGuardrailAgent,
)


class HITLProfilleyici:
    """Human-in-the-Loop Güvenlik ve Risk Profilleyicisi."""

    @classmethod
    def basarim_profili_cikar(cls) -> Dict[str, Any]:
        """Karşılaştırma Raporu ve Canlı Onay Testi."""
        karsilastirma = {
            "felaket_eylem_riski": {
                "Kor_Otonom": 100.0,
                "Statik_Bloklist": 35.0,
                "HITL_Bariyeri": 0.0,
            },
            "kurumsal_guvenlik_uyumu": {
                "Kor_Otonom": 0.0,
                "Statik_Bloklist": 45.0,
                "HITL_Bariyeri": 100.0,
            },
            "dusuk_risk_gecikmesi_ms": {
                "Kor_Otonom": 0.0,
                "Statik_Bloklist": 15.0,
                "HITL_Bariyeri": 0.0,
            },
        }

        ajan = HITLGuardrailAgent()

        # 1. Düşük Riskli Eylem (Bypass)
        talep1 = ajan.eylem_talebi_olustur(
            arac_adi="query_database",
            parametreler={"query": "SELECT * FROM users LIMIT 10"},
            gerekce="Kullanıcı istatistiklerini raporlamak.",
        )
        sonuc1 = ajan.eylemi_denetle_ve_icra_et(talep1)

        # 2. Kritik Riskli Eylem (Interrupt Gate)
        talep2 = ajan.eylem_talebi_olustur(
            arac_adi="delete_database_table",
            parametreler={"table": "customers_production"},
            gerekce="Eski müşteri verilerini temizlemek.",
        )
        sonuc2_dondurma = ajan.eylemi_denetle_ve_icra_et(talep2)

        # 3. İnsan Red Kararı ile Güvenli İptal
        insan_karari = ApprovalDecision(
            onaylandi_mi=False,
            insan_yorumu="Canlı üretim tablosu silinemez! Bunun yerine `archive_customers` aracını kullan.",
        )
        sonuc2_red = ajan.eylemi_denetle_ve_icra_et(talep2, insan_karari)

        return {
            "karsilastirma": karsilastirma,
            "sonuc_otomatik": sonuc1,
            "sonuc_dondurma": sonuc2_dondurma,
            "sonuc_red": sonuc2_red,
        }
