"""
PyTest Birim Testleri - Day 220: Grand Post-Training Benchmark Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark_motoru import (
    GSM8KEvaluator,
    MATH500Evaluator,
    HumanEvalEvaluator,
    MTBenchEvaluator,
    GrandBenchmarkSuite,
)
from src.faz11_sentez_profilleyici import Faz11SentezProfilleyici
from src.gorsellestirici import Faz11GrandBenchmarkGorsellestirici


def test_gsm8k_evaluator_valid():
    """1. GSM8KEvaluator sayısal cevabı doğru çıkarmalı ve doğrulamalıdır."""
    metin = "Toplam maliyet hesaplandı: 45 + 15 = 60. #### 60"
    assert GSM8KEvaluator.dogrula(metin, 60.0) is True
    assert GSM8KEvaluator.dogrula(metin, 50.0) is False


def test_math500_evaluator_boxed():
    """2. MATH500Evaluator LaTeX boxed cevabını doğru çıkarmalıdır."""
    metin = "İntegralin sonucu: \\boxed{e^x + C}"
    assert MATH500Evaluator.dogrula(metin, "e^x + C") is True
    assert MATH500Evaluator.dogrula(metin, "x^2") is False


def test_humaneval_evaluator_correct_code():
    """3. HumanEvalEvaluator çalışan Python fonksiyonunu başarıyla geçirmelidir."""
    kod = "def carp(a, b):\n    return a * b"
    test = "assert carp(4, 5) == 20"
    assert HumanEvalEvaluator.kod_test_et(kod, test) is True


def test_humaneval_evaluator_broken_code():
    """4. HumanEvalEvaluator hatalı Python kodunda False dönmelidir."""
    kod = "def carp(a, b):\n    return a + b"
    test = "assert carp(4, 5) == 20"
    assert HumanEvalEvaluator.kod_test_et(kod, test) is False


def test_mt_bench_evaluator_score_range():
    """5. MTBenchEvaluator geçerli bir kalite skoru (1-10) üretmelidir."""
    puan = MTBenchEvaluator.puanla("Detaylı açıklama 1.", "```python\npass\n```")
    assert 1.0 <= puan <= 10.0


def test_grand_benchmark_suite_execution():
    """6. GrandBenchmarkSuite tüm testleri başarıyla koşturmalıdır."""
    rapor = GrandBenchmarkSuite.tam_degerlendirme_kos()
    assert rapor["gsm8k_gecisi"] is True
    assert rapor["math500_gecisi"] is True
    assert rapor["humaneval_gecisi"] is True


def test_synthesis_profiler_metrics():
    """7. Sentez Profilleyicisi GRPO'nun taban modele göre açık ara üstün olduğunu göstermelidir."""
    sentez = Faz11SentezProfilleyici.sentez_raporu_cikar()
    gsm = sentez["metrikler"]["gsm8k"]
    assert gsm[-1] > gsm[0] + 40.0  # %92.4 vs %48.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. Faz11GrandBenchmarkGorsellestirici 6 panelli şampiyonluk panosunu üretmelidir."""
    cikti = str(tmp_path / "test_faz11_benchmark_paneli.png")
    sentez = Faz11SentezProfilleyici.sentez_raporu_cikar()

    Faz11GrandBenchmarkGorsellestirici.teshis_paneli_olustur(
        sentez_raporu=sentez,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
