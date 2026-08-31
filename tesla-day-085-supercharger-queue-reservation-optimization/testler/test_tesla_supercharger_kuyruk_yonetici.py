"""
Tesla Supercharger Kuyruk Birim Testleri (PyTest)
=================================================
Bu test paketi; M/M/c kuyruk modelini, aşırı yük tespiti durumunu
ve FSD dinamik alternatif istasyon yönlendirmesini test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_supercharger_kuyruk_yonetici import TeslaSuperchargerQueueManager


def test_mmc_normal_trafik_analizi():
    """12 stall ve 25 araç/saat varışta bekleme süresinin makul (<5 dk) olduğu test edilir."""
    mgr = TeslaSuperchargerQueueManager(num_stalls=12, service_rate_per_stall_per_hour=3.0)
    res = mgr.calculate_mmc_metrics(arrival_rate_lambda=25.0)

    assert res["is_stable"] is True
    assert res["utilization_rho"] < 1.0
    assert res["avg_wait_time_mins"] < 5.0
    assert res["reroute_recommended"] is False


def test_mmc_asiri_yuk_tespiti():
    """Kapasiteyi aşan varış hızında (lambda >= 36) sistemin kararsız olduğu test edilir."""
    mgr = TeslaSuperchargerQueueManager(num_stalls=12, service_rate_per_stall_per_hour=3.0)
    res = mgr.calculate_mmc_metrics(arrival_rate_lambda=38.0)

    assert res["is_stable"] is False
    assert res["utilization_rho"] >= 1.0
    assert res["reroute_recommended"] is True


def test_fsd_alternatif_istasyon_yonlendirme():
    """Aşırı bekleme süresi olan istasyondan alternatif istasyona yönlendirildiği test edilir."""
    mgr = TeslaSuperchargerQueueManager(num_stalls=12, service_rate_per_stall_per_hour=3.0, max_acceptable_wait_mins=10.0)
    # Yüksek yoğunluk (lambda=35.0)
    res = mgr.evaluate_fsd_route_reservation(current_arrival_rate=35.0, eta_minutes=12.0, alternate_station_wait_mins=2.0)

    assert res["decision"] == "REROUTE_TO_ALTERNATE_STATION"
    assert res["assigned_wait_mins"] == 2.0
