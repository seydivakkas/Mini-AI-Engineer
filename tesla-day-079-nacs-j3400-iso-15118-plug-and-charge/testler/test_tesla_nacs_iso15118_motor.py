"""
Tesla NACS ve ISO 15118 Birim Testleri (PyTest)
===============================================
Bu test paketi; Control Pilot voltaj durumlarını,
ISO 15118 sözleşme sertifikası doğrulamasını ve V2G mesaj üretimini test eder.

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

from src.tesla_nacs_iso15118_motor import TeslaNACSISO15118Engine, ControlPilotState


def test_soket_baglantisi_ve_cp_durumu():
    """Soket takıldığında CP'nin State B'ye geçtiği test edilir."""
    engine = TeslaNACSISO15118Engine()
    res = engine.handle_plug_connection()

    assert res["cp_state"] == ControlPilotState.STATE_B_CONNECTED.value
    assert res["cp_voltage_v"] == 9.0
    assert res["pwm_duty_pct"] == 5.0


def test_iso15118_sozlesme_dogrulama():
    """Geçerli araç VIN ve sertifikasıyla State C şarj durumuna geçildiği test edilir."""
    engine = TeslaNACSISO15118Engine()
    engine.handle_plug_connection()

    res = engine.verify_iso15118_contract(
        vehicle_vin="5YJ3E1EB8NF123456",
        contract_token="CONTRACT_TESLA_VIP",
        oem_signature="SIG_ECDSA_VALID_KEY"
    )

    assert res["contract_verified"] is True
    assert res["authorization_status"] == "ACCEPTED"
    assert res["cp_state"] == ControlPilotState.STATE_C_CHARGING.value


def test_v2g_akim_talebi_mesaji():
    """State C durumunda geçerli CurrentDemandReq mesajı üretildiği test edilir."""
    engine = TeslaNACSISO15118Engine()
    engine.handle_plug_connection()
    engine.verify_iso15118_contract("5YJ3E1EB8NF123456", "CONTRACT_TESLA_VIP", "SIG_ECDSA_VALID_KEY")

    v2g_msg = engine.create_v2g_charge_loop_message(target_voltage_v=400.0, max_current_a=500.0, soc_pct=50.0)

    assert v2g_msg["msg_type"] == "CurrentDemandReq"
    assert v2g_msg["charging_power_kw"] == 200.0
    assert v2g_msg["contactor_closed"] is True
