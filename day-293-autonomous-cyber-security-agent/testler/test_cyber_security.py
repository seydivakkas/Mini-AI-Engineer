"""
PyTest Birim Testleri - Day 293 (FAZ 15): Otonom Siber Güvenlik ve Zero-Day Savunma.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cyber_security_motoru import (
    Vulnerability,
    VulnerabilityScanner,
    SandboxExploitTester,
    AutoPatchGenerator,
)
from src.cyber_security_profilleyici import CyberSecurityProfilleyici
from src.gorsellestirici import CyberSecurityGorsellestirici


def test_vulnerability_initialization():
    """1. Zafiyet modeli CVE kimliği, CVSS puanı ve durum bilgisiyle başlatılmalıdır."""
    vuln = Vulnerability("SQLi Test", "CVE-2026-9999", 9.8, "execute(sql)")
    assert vuln.name == "SQLi Test"
    assert vuln.cvss_score == 9.8
    assert not vuln.is_patched


def test_vulnerability_scanner_detection():
    """2. Zafiyet tarayıcısı güvensiz SQL birleştirmelerini tespit etmelidir."""
    code = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
    vulns = VulnerabilityScanner.scan_codebase(code)
    assert len(vulns) >= 1
    assert vulns[0].category == "SQL_INJECTION"


def test_sandbox_exploit_pre_patch():
    """3. Yamalanmamış kod kum havuzunda exploit testinde başarısız (savunmasız) olmalıdır."""
    vuln = Vulnerability("SQLi", "CVE-TEST", 9.8, "raw code")
    is_exploitable = SandboxExploitTester.test_exploit(vuln, "admin' OR '1'='1")
    assert is_exploitable is True


def test_auto_patch_generator_application():
    """4. Otomatik yama üretici parametrik sorgu üretmeli ve yamayı onaylamalıdır."""
    vuln = Vulnerability("SQLi", "CVE-TEST", 9.8, "raw code")
    patched = AutoPatchGenerator.generate_and_apply_patch(vuln)
    assert "%s" in patched
    assert vuln.is_patched is True


def test_sandbox_exploit_post_patch():
    """5. Yamalanmış kod kum havuzunda exploit girişimlerini engellemelidir."""
    vuln = Vulnerability("SQLi", "CVE-TEST", 9.8, "raw code")
    AutoPatchGenerator.generate_and_apply_patch(vuln)
    is_exploitable = SandboxExploitTester.test_exploit(vuln, "admin' OR '1'='1")
    assert is_exploitable is False


def test_profiler_mttr_speedup():
    """6. MTTR onarım süresi hızlanma oranı 10,000 kattan fazla olmalıdır."""
    profil = CyberSecurityProfilleyici.basarim_profili_cikar()
    assert profil["hizlanma_orani"] >= 10000.0


def test_profiler_zero_day_accuracy():
    """7. Otonom savunma ajanı zero-day tespit oranı %95'in üzerinde olmalıdır."""
    profil = CyberSecurityProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["zero_day_tespit_orani_yuzde"]["3. Autonomous Defense"] > 95.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. CyberSecurityGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_cyber_paneli.png")
    profil = CyberSecurityProfilleyici.basarim_profili_cikar()

    CyberSecurityGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
