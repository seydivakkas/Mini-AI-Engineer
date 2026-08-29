"""
PyTest Birim Testleri - Day 228: SWE-Bench Otonom Kodlayıcı Ajan Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.swe_kodlayici_motoru import (
    GitHubIssue,
    CodebaseNavigator,
    SurgicalPatcher,
    AutonomousSWEAgent,
)
from src.swe_profilleyici import SWEProfilleyici
from src.gorsellestirici import SWEGorsellestirici


def test_github_issue_initialization():
    """1. GitHubIssue veri modeli tüm alanları doğru başlatmalıdır."""
    issue = GitHubIssue(101, "Bug", "Desc", "Trace", "file.py")
    assert issue.issue_id == 101
    assert issue.target_file == "file.py"


def test_codebase_navigator_slice():
    """2. CodebaseNavigator satır aralığını numaralandırarak kesit olarak dönmelidir."""
    kod = "satir1\nsatir2\nsatir3\nsatir4"
    kesit = CodebaseNavigator.dosya_kesiti_oku(kod, 2, 3)
    assert "2: satir2" in kesit
    assert "3: satir3" in kesit
    assert "1: satir1" not in kesit


def test_surgical_patch_application():
    """3. SurgicalPatcher dosyayı bozmadan sadece hedef bloğu yamalamalıdır."""
    kod = "a = 1\nb = 2\nc = 3\n"
    yamalanmis, diff = SurgicalPatcher.cerrahi_yama_uygula(kod, "b = 2", "b = 20")
    assert "b = 20" in yamalanmis
    assert "a = 1" in yamalanmis
    assert "c = 3" in yamalanmis


def test_surgical_patch_unified_diff():
    """4. SurgicalPatcher geçerli Unified Git Diff formatı üretmelidir."""
    kod = "x = 10\n"
    _, diff = SurgicalPatcher.cerrahi_yama_uygula(kod, "x = 10", "x = 20")
    assert "--- a/dosya.py" in diff
    assert "+++ b/dosya.py" in diff
    assert "- x = 10" in diff
    assert "+ x = 20" in diff


def test_surgical_patch_missing_chunk_error():
    """5. SurgicalPatcher hedef blok bulunamadığında ValueError fırlatmalıdır."""
    kod = "x = 10\n"
    with pytest.raises(ValueError):
        SurgicalPatcher.cerrahi_yama_uygula(kod, "y = 99", "y = 100")


def test_autonomous_swe_agent_solve_flow():
    """6. AutonomousSWEAgent sorunu başarıyla çözüp diff üretmelidir."""
    issue = GitHubIssue(1, "Fix", "Zero div", "Trace", "math.py")
    kod = "return 1 / x"
    ajan = AutonomousSWEAgent()
    sonuc = ajan.sorunu_coz_ve_yamala(issue, kod, "return 1 / x", "if x == 0: return 0\nreturn 1 / x")
    assert sonuc["cozum_basarili_mi"] is True
    assert "Unified Git Diff" in str(sonuc["islem_adimlari"])


def test_profiler_swe_metrics():
    """7. Profilleyici SWE-Bench otonom ajanının çözüm oranının %50 üstünde olduğunu göstermelidir."""
    prof = SWEProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["swe_bench_cozum_orani"]["SWE_Bench_Otonom_Ajan"]
    assert skor > 50.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SWEGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_swe_paneli.png")
    profil = SWEProfilleyici.basarim_profili_cikar()

    SWEGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
