"""
PyTest Birim Testleri - Day 232: Human-in-the-Loop (HITL) Güvenlik Bariyeri Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hitl_motoru import (
    RiskLevel,
    ActionRequest,
    ApprovalDecision,
    HITLGuardrailAgent,
)
from src.hitl_profilleyici import HITLProfilleyici
from src.gorsellestirici import HITLGorsellestirici


def test_risk_level_enum_values():
    """1. RiskLevel enum tüm seviyeleri doğru tanımlamalıdır."""
    assert RiskLevel.LOW.value == "DUSUK"
    assert RiskLevel.CRITICAL.value == "KRITIK"


def test_action_request_initialization():
    """2. ActionRequest nesnesi eylem ve risk alanlarını doğru başlatmalıdır."""
    talep = ActionRequest("read_file", {"path": "a.txt"}, RiskLevel.LOW, "okuma")
    assert talep.arac_adi == "read_file"
    assert talep.risk_seviyesi == RiskLevel.LOW


def test_approval_decision_initialization():
    """3. ApprovalDecision varsayılan açıklamaları doğru atamalıdır."""
    karar_onay = ApprovalDecision(True)
    karar_red = ApprovalDecision(False)
    assert "onaylandı" in karar_onay.insan_yorumu
    assert "reddedildi" in karar_red.insan_yorumu


def test_risk_classification_low_risk():
    """4. HITLGuardrailAgent sorgu ve okuma işlemlerini DÜŞÜK risk olarak belirlemelidir."""
    risk = HITLGuardrailAgent.risk_derecelendir("query_database", {})
    assert risk == RiskLevel.LOW


def test_risk_classification_critical_risk():
    """5. HITLGuardrailAgent silme ve transfer işlemlerini KRİTİK risk olarak belirlemelidir."""
    risk = HITLGuardrailAgent.risk_derecelendir("delete_database_table", {})
    assert risk == RiskLevel.CRITICAL


def test_hitl_auto_execution_on_low_risk():
    """6. Düşük riskli işlemler insan onayına gerek kalmadan doğrudan icra edilmelidir."""
    ajan = HITLGuardrailAgent()
    talep = ajan.eylem_talebi_olustur("read_file", {"path": "test.txt"}, "okuma")
    sonuc = ajan.eylemi_denetle_ve_icra_et(talep)
    assert sonuc["durum"] == "OTOMATIK_ICRA"
    assert sonuc["icra_edildi_mi"] is True


def test_hitl_interrupt_and_human_approval():
    """7. Kritik riskli işlemler onaysızken INTERRUPT edilmeli, onay verildiğinde icra edilmelidir."""
    ajan = HITLGuardrailAgent()
    talep = ajan.eylem_talebi_olustur("delete_file", {"path": "prod.db"}, "silme")

    # Onaysız -> Dondur
    dondurma = ajan.eylemi_denetle_ve_icra_et(talep)
    assert dondurma["durum"] == "INTERRUPT_BEKLEMEDE"
    assert dondurma["icra_edildi_mi"] is False

    # Onaylı -> İcra Et
    onay = ApprovalDecision(True, "Onaylandı")
    icra = ajan.eylemi_denetle_ve_icra_et(talep, onay)
    assert icra["durum"] == "ONAYLI_ICRA"
    assert icra["icra_edildi_mi"] is True


def test_hitl_rejection_and_visualizer(tmp_path):
    """8. İnsan reddi eylemi durdurmalı ve 6 panelli teşhis panosu başarıyla üretilmelidir."""
    ajan = HITLGuardrailAgent()
    talep = ajan.eylem_talebi_olustur("transfer_funds", {"amount": 10000}, "transfer")
    red = ApprovalDecision(False, "İzin verilmedi")
    sonuc = ajan.eylemi_denetle_ve_icra_et(talep, red)
    assert sonuc["durum"] == "REDDEDILDI"
    assert sonuc["icra_edildi_mi"] is False

    cikti = str(tmp_path / "test_hitl_paneli.png")
    profil = HITLProfilleyici.basarim_profili_cikar()
    HITLGorsellestirici.teshis_paneli_olustur(profil, kayit_yolu=cikti)
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
