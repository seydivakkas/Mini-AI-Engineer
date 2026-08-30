"""
Day 360: Aerospace, Defense & Deep Space Autonomous AI Operating System (AeroSpace-AI-OS)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aerospace_ai_os_motoru import (
    SubsystemTaskPriority,
    MissionPhaseState,
    RTOSRealTimeScheduler,
    FaultTolerantSubsystemManager,
    AeroSpaceAutonomousAIOS,
)
from src.os_profilleyici import OSProfilleyici


def test_rtos_real_time_scheduler_dispatch():
    """
    RTOS Öncelikli Görev Zamanlayıcı Testi.
    """
    sched = RTOSRealTimeScheduler(max_allowed_latency_ms=2.0)
    sched.submit_task("GNC", SubsystemTaskPriority.CRITICAL_FLIGHT_GNC, 0.2)
    sched.submit_task("Telemetry", SubsystemTaskPriority.BACKGROUND_TELEMETRY, 0.1)
    
    res = sched.dispatch_cycle()
    assert len(res) == 2
    assert res[0]["priority"] == "CRITICAL_FLIGHT_GNC"
    assert res[0]["deadline_met"] is True


def test_fault_tolerant_subsystem_manager_tmr():
    """
    TMR 2/3 Oylama ve SEU Düzeltim Testi.
    """
    mgr = FaultTolerantSubsystemManager()
    val, corr = mgr.verify_tmr_execution(command_val=0x55, inject_fault=True)
    assert val == 0x55
    assert corr is True
    assert mgr.total_seu_corrected == 1


def test_aerospace_ai_os_full_cycle():
    """
    Tam Entegre AeroSpace-AI-OS Görev Döngüsü Testi.
    """
    os_sys = AeroSpaceAutonomousAIOS()
    res = os_sys.execute_mission_cycle(steps=10)
    assert res["total_tasks_executed"] == 50
    assert res["deadline_success_rate"] == 100.0
    assert res["os_healthy"] is True


def test_os_profiler_metrics():
    """
    İşletim Sistemi Profilleyici Metrik Testi.
    """
    mock_res = {
        "deadline_success_rate": 100.0,
        "seu_recovery_rate": 100.0
    }
    metrics = OSProfilleyici.profille(mock_res)
    assert metrics["rtos_deadline_score"] == 100.0
    assert metrics["fault_tolerance_score"] == 100.0
    assert metrics["os_readiness_score"] > 99.0
