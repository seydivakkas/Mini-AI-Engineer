"""
PyTest Birim Testleri - Day 239: GAIA Ajan Benchmark Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gaia_benchmark_motoru import (
    GAIATask,
    GAIAEvaluator,
    GAIAAgentHarness,
)
from src.gaia_profilleyici import GAIAProfilleyici
from src.gorsellestirici import GAIAGorsellestirici


def test_gaia_task_initialization():
    """1. GAIATask alanları doğru başlatmalıdır."""
    task = GAIATask("g-1", 1, "Soru", "100", ["Search"])
    assert task.task_id == "g-1"
    assert task.level == 1
    assert task.dogru_mu is False


def test_gaia_evaluator_exact_match_string():
    """2. GAIAEvaluator metin eşleşmelerini doğru doğrulamalıdır."""
    assert GAIAEvaluator.dogrula("Ankara", "ankara") is True
    assert GAIAEvaluator.dogrula("Istanbul", "Ankara") is False


def test_gaia_evaluator_numerical_tolerance():
    """3. GAIAEvaluator sayısal değerlerde %1 toleransı doğru hesaplamalıdır."""
    assert GAIAEvaluator.dogrula("100.5", "100.0", tolerans=0.01) is True
    assert GAIAEvaluator.dogrula("110.0", "100.0", tolerans=0.01) is False


def test_gaia_evaluator_normalization():
    """4. GAIAEvaluator para birimi ve boşlukları normalize etmelidir."""
    assert GAIAEvaluator.dogrula("$1,000,000", "1000000") is True


def test_gaia_harness_pool_creation():
    """5. GAIAAgentHarness 4 görevli örnek havuzu eksiksiz yüklemelidir."""
    h = GAIAAgentHarness()
    h.ornek_gaia_havuzu_olustur()
    assert len(h.gorevler) == 4


def test_gaia_harness_evaluation_report():
    """6. GAIAAgentHarness seviye bazlı karne ve genel skoru hesaplamalıdır."""
    h = GAIAAgentHarness()
    h.ornek_gaia_havuzu_olustur()
    tahminler = {"gaia-101": "3", "gaia-102": "4500000", "gaia-201": "150000", "gaia-301": "128450.5"}
    rapor = h.degerlendir(tahminler)
    assert rapor["genel_gaia_skoru"] == 100.0
    assert rapor["seviye_1_basari"] == 100.0


def test_profiler_gaia_metrics():
    """7. Profilleyici GAIA Ajan başarısının %70 üzerinde olduğunu doğrulamalıdır."""
    prof = GAIAProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["genel_gaia_skoru"]["Cok_Modlu_GAIA_Ajani"]
    assert skor > 70.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. GAIAGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_gaia_paneli.png")
    profil = GAIAProfilleyici.basarim_profili_cikar()

    GAIAGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
