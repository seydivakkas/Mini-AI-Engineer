"""
PyTest Birim Testleri - Day 240: Otonom Ajan Süiti (Agentic AI OS - FAZ 12 FİNALİ) Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.platform_ajani_motoru import AgenticAIPlatform
from src.platform_profilleyici import PlatformProfilleyici
from src.gorsellestirici import PlatformGorsellestirici


def test_platform_subsystem_health():
    """1. AgenticAIPlatform tüm alt sistemleri AKTİF olarak başlatmalıdır."""
    p = AgenticAIPlatform()
    for k, v in p.sistem_durumu.items():
        assert v == "AKTİF"


def test_platform_plan_and_solve_decomposition():
    """2. AgenticAIPlatform kök görevi 5 adımlı WBS planına ayrıştırmalıdır."""
    p = AgenticAIPlatform()
    plan = p.plan_ve_ayristir("Finansal Rapor")
    assert len(plan) == 5


def test_platform_hitl_low_risk_auto_pass():
    """3. HITL düşük riskli okuma işlemlerini doğrudan geçirmelidir."""
    p = AgenticAIPlatform()
    res = p.hitl_risk_kontrol("READ")
    assert res["durum"] == "OTOMATIK_GECTI"
    assert res["risk"] == "LOW"


def test_platform_hitl_critical_approved():
    """4. HITL onaylanan kritik işlemde ONAYLANDI dönmelidir."""
    p = AgenticAIPlatform()
    res = p.hitl_risk_kontrol("DEPLOY", insan_onayi=True)
    assert res["durum"] == "ONAYLANDI"
    assert res["risk"] == "CRITICAL"


def test_platform_hitl_critical_rejected():
    """5. HITL reddedilen kritik işlemde boru hattını durdurmalıdır."""
    p = AgenticAIPlatform()
    res = p.tam_is_akisi_yurut("Riskli Silme", kritik_eylem_var_mi=True, insan_onayi=False)
    assert res["basarili_mi"] is False
    assert res["durum"] == "HITL_ENGELLEDİ"


def test_platform_full_workflow_execution():
    """6. AgenticAIPlatform tam uçtan uca akışı başarıyla tamamlamalıdır."""
    p = AgenticAIPlatform()
    res = p.tam_is_akisi_yurut("Tam Otomasyon", kritik_eylem_var_mi=False)
    assert res["basarili_mi"] is True
    assert res["durum"] == "TAMAMLANDI"
    assert res["kalite_skoru"] >= 90.0


def test_profiler_capstone_metrics():
    """7. Profilleyici Agentic AI OS başarısının %90 üzerinde olduğunu doğrulamalıdır."""
    prof = PlatformProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["uctan_uca_gorev_basarisi"]["Agentic_AI_OS"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PlatformGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_capstone_paneli.png")
    profil = PlatformProfilleyici.basarim_profili_cikar()

    PlatformGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
