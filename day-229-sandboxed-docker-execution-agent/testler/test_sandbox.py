"""
PyTest Birim Testleri - Day 229: Güvenli Docker Sandbox Ajan Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sandbox_motoru import (
    SandboxConfig,
    ExecutionResult,
    DockerSandboxAgent,
)
from src.sandbox_profilleyici import SandboxProfilleyici
from src.gorsellestirici import SandboxGorsellestirici


def test_sandbox_config_defaults():
    """1. SandboxConfig nesnesi varsayılan limitleri doğru başlatmalıdır."""
    cfg = SandboxConfig()
    assert cfg.bellek_limiti_mb == 512
    assert "os.system" in cfg.yasakli_kalıplar


def test_execution_result_formatting():
    """2. ExecutionResult nesnesi çıktıları doğru formatlamalıdır."""
    res = ExecutionResult("Test Sonucu", "", 0, 1.5)
    assert "STDOUT: Test Sonucu" in res.format_metni()
    assert "Exit: 0" in res.format_metni()


def test_safe_code_execution():
    """3. DockerSandboxAgent güvenli Python kodunu çalıştırıp çıktıyı yakalamalıdır."""
    ajan = DockerSandboxAgent()
    res = ajan.kodu_izole_calistir("print('Merhaba Sandbox')")
    assert res.exit_code == 0
    assert "Merhaba Sandbox" in res.stdout
    assert res.guvenlik_ihlali is False


def test_runtime_error_capture():
    """4. DockerSandboxAgent çalışma zamanı hatasını stderr'de yakalamalıdır."""
    ajan = DockerSandboxAgent()
    res = ajan.kodu_izole_calistir("x = 1 / 0")
    assert res.exit_code == 1
    assert "division by zero" in res.stderr


def test_security_policy_blocking_os_system():
    """5. DockerSandboxAgent os.system çağrısını çalıştırmadan bloke etmelidir."""
    ajan = DockerSandboxAgent()
    res = ajan.kodu_izole_calistir("import os; os.system('echo test')")
    assert res.guvenlik_ihlali is True
    assert res.exit_code == 126
    assert "Yasaklı" in str(res.ihlal_mesaji)


def test_security_policy_blocking_subprocess():
    """6. DockerSandboxAgent subprocess çağrısını bloke etmelidir."""
    ajan = DockerSandboxAgent()
    res = ajan.kodu_izole_calistir("import subprocess; subprocess.run(['ls'])")
    assert res.guvenlik_ihlali is True
    assert res.exit_code == 126


def test_profiler_sandbox_metrics():
    """7. Profilleyici Docker Sandbox mimarisinde host riskinin %0 olduğunu doğrulamalıdır."""
    prof = SandboxProfilleyici.basarim_profili_cikar()
    risk = prof["karsilastirma"]["ana_sistem_guvenlik_riski"]["Docker_Sandbox"]
    assert risk == 0.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SandboxGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_sandbox_paneli.png")
    profil = SandboxProfilleyici.basarim_profili_cikar()

    SandboxGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
