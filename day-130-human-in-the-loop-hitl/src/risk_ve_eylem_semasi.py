"""
Risk Sınıflandırma ve Ajan Eylem Şeması Modülü (Day 130 - Faz 7).
Eylem risk skorlama, tehlike seviyeleri (Düşük, Orta, Yüksek, Kritik) ve Human-in-the-Loop onay kuralları.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class EylemSeviyesi(Enum):
    DUSUK = "DUSUK"       # Risk < 0.30: Otomatik icra edilir
    ORTA = "ORTA"         # 0.30 <= Risk < 0.70: Loglanır ve otomatik icra edilir
    YUKSEK = "YUKSEK"     # 0.70 <= Risk < 0.90: İnsan onayı için kesinti (Interrupt)
    KRITIK = "KRITIK"     # Risk >= 0.90: İki kademeli yetkili denetçi onayı şarttır


@dataclass
class AjanEylemi:
    """Ajan tarafından planlanan tek bir operasyon."""
    eylem_id: str
    eylem_adi: str
    parametreler: Dict[str, Any]
    risk_skoru: float = 0.0
    seviye: EylemSeviyesi = EylemSeviyesi.DUSUK
    onay_gerekli_mi: bool = False
    insan_karari: Optional[str] = None  # None, "ONAYLANDI", "REDDEDILDI", "DUZENLENDI"
    yurutuldu_mu: bool = False
    sonuc_mesaji: str = ""


class RiskSiniflandirici:
    """Ajan eylemlerini analiz eden ve risk skoru atayan güvenlik motoru."""

    TEHLIKELI_EYLEMLER = {
        "veritabani_tablo_sil": (0.95, EylemSeviyesi.KRITIK),
        "sunucu_kapat": (0.85, EylemSeviyesi.YUKSEK),
        "dns_yonlendirme_degistir": (0.88, EylemSeviyesi.YUKSEK),
        "toplu_eposta_gonder": (0.75, EylemSeviyesi.YUKSEK),
        "para_transferi": (0.50, EylemSeviyesi.ORTA),  # Tutara göre artar
        "log_sorgula": (0.05, EylemSeviyesi.DUSUK),
        "rapor_olustur": (0.10, EylemSeviyesi.DUSUK),
    }

    @classmethod
    def eylemi_degerlendir(cls, eylem_adi: str, parametreler: Dict[str, Any]) -> AjanEylemi:
        """Eylemin risk skorunu ve onay gereksinimini dinamik olarak hesaplar."""
        baz_risk, seviye = cls.TEHLIKELI_EYLEMLER.get(eylem_adi, (0.40, EylemSeviyesi.ORTA))
        hesaplanan_risk = baz_risk

        # Dinamik Parametreye Dayalı Risk Artırımı
        if eylem_adi == "para_transferi":
            tutar = parametreler.get("tutar", 0.0)
            if tutar > 50000.0:
                hesaplanan_risk = 0.92
                seviye = EylemSeviyesi.KRITIK
            elif tutar > 5000.0:
                hesaplanan_risk = 0.78
                seviye = EylemSeviyesi.YUKSEK

        onay_gerekli = seviye in [EylemSeviyesi.YUKSEK, EylemSeviyesi.KRITIK]

        return AjanEylemi(
            eylem_id=f"ACT_{abs(hash(eylem_adi + str(parametreler))) % 100000}",
            eylem_adi=eylem_adi,
            parametreler=parametreler,
            risk_skoru=hesaplanan_risk,
            seviye=seviye,
            onay_gerekli_mi=onay_gerekli,
        )
