"""
Tesla Büyük Final ve Portföy Gezgini Birim Testleri (PyTest)
===========================================================
Bu test paketi; 11 haftalık müfredatı, yönetici özetini ve mezuniyet sertifikasını test eder.

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

from src.tesla_portfoy_gezgini import TeslaPortfolioNavigator


def test_onbir_hafta_ve_99_gun_kapsami():
    """11 haftanın ve 99 günün eksiksiz kapsandığı test edilir."""
    nav = TeslaPortfolioNavigator()
    weeks = nav.get_weekly_curriculum()

    assert len(weeks) == 11
    assert weeks[0].day_start == 1
    assert weeks[-1].day_end == 99


def test_yonetici_ozeti_ve_sertifika_dogrulama():
    """Yönetici özeti ve Grandmaster sertifikası doğrulanır."""
    nav = TeslaPortfolioNavigator()
    exec_sum = nav.generate_executive_summary()
    cert = nav.generate_graduation_certificate()

    assert exec_sum["total_days_completed"] == 99
    assert exec_sum["total_test_pass_rate_pct"] == 100.0
    assert "Seydi Eryılmaz" in cert["recipient"]
    assert "PRINCIPAL" in cert["degree_awarded"]
    assert "SUMMA CUM LAUDE" in cert["honors"]
