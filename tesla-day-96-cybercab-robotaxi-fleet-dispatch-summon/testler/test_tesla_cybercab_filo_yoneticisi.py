"""
Tesla Cybercab Filo Yönetimi Birim Testleri (PyTest)
====================================================
Bu test paketi; Cybercab otonom yolcu eşleştirmesini, ETA optimizasyonunu
ve batarya dengeleme şarj yönlendirmesini test eder.

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

from src.tesla_cybercab_filo_yoneticisi import TeslaCybercabFleetDispatcher, CybercabVehicle, PassengerRequest


def test_cybercab_yolcu_eslestirme():
    """En yakın ve şarjı yeterli olan Cybercab'in seçildiği test edilir."""
    dispatcher = TeslaCybercabFleetDispatcher(avg_speed_kmh=60.0, min_soc_for_trip=20.0)

    fleet = [
        CybercabVehicle(cab_id="CAB_01", x_km=10.0, y_km=10.0, soc_pct=80.0, status="AVAILABLE"),
        CybercabVehicle(cab_id="CAB_02", x_km=2.0, y_km=2.0, soc_pct=15.0, status="AVAILABLE"),  # Şarjı az
        CybercabVehicle(cab_id="CAB_03", x_km=1.0, y_km=1.0, soc_pct=90.0, status="AVAILABLE"),   # En yakın ve şarjı tam!
    ]

    req = PassengerRequest(req_id="R1", pickup_x_km=0.0, pickup_y_km=0.0, dest_x_km=5.0, dest_y_km=5.0)

    res = dispatcher.dispatch_trip(req, fleet)

    assert res["matched"] is True
    assert res["assigned_cab_id"] == "CAB_03"
    assert res["eta_minutes"] < 5.0


def test_musait_arac_olmama_durumu():
    """Tüm araçlar yolculukta veya şarjsız ise çağrının güvenle reddedildiği test edilir."""
    dispatcher = TeslaCybercabFleetDispatcher(min_soc_for_trip=20.0)
    fleet = [
        CybercabVehicle(cab_id="CAB_01", x_km=1.0, y_km=1.0, soc_pct=10.0, status="AVAILABLE"),
        CybercabVehicle(cab_id="CAB_02", x_km=2.0, y_km=2.0, soc_pct=80.0, status="ON_TRIP"),
    ]
    req = PassengerRequest(req_id="R2", pickup_x_km=0.0, pickup_y_km=0.0, dest_x_km=5.0, dest_y_km=5.0)

    res = dispatcher.dispatch_trip(req, fleet)
    assert res["matched"] is False
    assert "NO_AVAILABLE_CAB" in res["reason"]


def test_otonom_sarj_dengeleme():
    """%20 altındaki araçların otomatik şarj durumuna geçirildiği test edilir."""
    dispatcher = TeslaCybercabFleetDispatcher(min_soc_for_trip=20.0)
    fleet = [
        CybercabVehicle(cab_id="CAB_01", x_km=0.0, y_km=0.0, soc_pct=12.0, status="AVAILABLE"),
        CybercabVehicle(cab_id="CAB_02", x_km=0.0, y_km=0.0, soc_pct=75.0, status="AVAILABLE"),
    ]

    routed = dispatcher.auto_supercharge_rebalancing(fleet)
    assert "CAB_01" in routed
    assert "CAB_02" not in routed
    assert fleet[0].status == "CHARGING"
