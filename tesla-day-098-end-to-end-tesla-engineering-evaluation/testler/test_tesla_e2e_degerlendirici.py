"""
Tesla E2E Şampiyonluk Değerlendiricisi Birim Testleri (PyTest)
=============================================================
Bu test paketi; 8 temel mühendislik sütununu ve şampiyonluk skorunu test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_e2e_degerlendirici import TeslaE2EEngineeringEvaluator


def test_sekiz_sutun_tam_degerlendirme():
    """8 temel sütunun eksiksiz değerlendirildiği test edilir."""
    evaluator = TeslaE2EEngineeringEvaluator()
    pillars = evaluator.evaluate_all_pillars()

    assert len(pillars) == 8
    total_weight = sum(p.weight for p in pillars)
    assert abs(total_weight - 1.00) < 1e-5


def test_sampiyonluk_skoru_ve_unvan_atama():
    """Şampiyonluk skorunun %100 ve unvanın Principal Architect olduğu test edilir."""
    evaluator = TeslaE2EEngineeringEvaluator()
    pillars = evaluator.evaluate_all_pillars()
    res = evaluator.calculate_championship_score(pillars)

    assert res["total_championship_score"] == 100.0
    assert res["all_pillars_passed"] is True
    assert res["is_tesla_grandmaster"] is True
    assert "PRINCIPAL" in res["title_awarded"]
    assert "SUMMA CUM LAUDE" in res["certification_status"]
