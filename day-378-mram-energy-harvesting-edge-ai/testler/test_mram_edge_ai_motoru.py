"""
Day 378: Unit Tests for Energy-Harvesting STT-MRAM Edge AI Accelerator
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mram_edge_ai_motoru import (
    AmbientEnergyHarvester,
    STTMRAMArray,
    IntermittentTinyMLEngine,
    MRAMEdgeAIBenchmark,
    PowerState
)


def test_stt_mram_tmr_ve_fiziksel_direncler():
    """STT-MRAM MTJ dirençlerinin ve TMR manyeto-direnç oranının doğruluğunu test eder."""
    mram = STTMRAMArray(rows=16, cols=16, rp_kohm=1.0, rap_kohm=2.5)
    
    assert mram.rp == 1000.0, "Paralel direnç 1000 Ohm olmalıdır."
    assert mram.rap == 2500.0, "Anti-paralel direnç 2500 Ohm olmalıdır."
    # TMR = (2500 - 1000) / 1000 * 100 = %150
    assert abs(mram.tmr - 150.0) < 1e-4, "TMR oranı %150 olmalıdır."
    assert mram.standby_leakage_nw == 0.0, "MRAM statik sızıntısı 0.0 nW olmalıdır."


def test_enerji_hasadi_ve_kapasitor_sarji():
    """Kapasitör voltaj dinamiğinin ve brownout tespit eşiklerinin doğruluğunu test eder."""
    harvester = AmbientEnergyHarvester(capacitance_uf=100.0, v_operating_min=1.8, v_brownout_th=2.0)
    
    # 1. Yüksek güç girişi -> Voltaj artmalı ve INFERENCE durumuna geçmeli
    v_cap, state = harvester.step_harvesting(p_harvest_uw=200.0, p_consume_uw=20.0, dt_ms=10.0)
    assert v_cap > 1.8, "Kapasitör şarj olmalıdır."
    assert state in [PowerState.INFERENCE, PowerState.CHECKPOINTING]

    # 2. Güç kesintisi ve deşarj -> SLEEP_ZERO_POWER durumuna geçmeli
    harvester.v_cap = 1.0  # Min seviyenin altına düşür
    v_cap2, state2 = harvester.step_harvesting(p_harvest_uw=0.0, p_consume_uw=0.0, dt_ms=1.0)
    assert state2 == PowerState.SLEEP_ZERO_POWER


def test_kesintili_checkpoint_ve_durum_kurtarma():
    """Uçucu olmayan MRAM bellek üzerinde anında durum kaydetme ve geri yüklemeyi test eder."""
    mram = STTMRAMArray(rows=8, cols=8)
    dummy_state = {"progress": 1, "tensor_val": 42.0}
    
    e_save = mram.save_checkpoint(dummy_state)
    assert e_save > 0.0, "Checkpoint kaydetme enerjisi harcanmalıdır."

    restored_state, e_restore = mram.restore_checkpoint()
    assert restored_state["progress"] == 1
    assert restored_state["tensor_val"] == 42.0
    assert e_restore > 0.0


def test_tam_edge_ai_benchmark_ve_sifir_sizinti():
    """Tam 300 ms kesintili ortam simülasyonunu ve MRAM sıfır sızıntı avantajını test eder."""
    bench = MRAMEdgeAIBenchmark()
    res = bench.kos(num_steps=100)

    assert res["completed_inferences"] > 0, "En az 1 çıkarım tamamlanmalıdır."
    assert res["sram_leakage_uj"] > 0.0, "SRAM statik enerji sızdırmalıdır."
    assert res["mram_leakage_uj"] == 0.0, "STT-MRAM statik sızıntısı 0.0 uJ olmalıdır."
    assert res["forward_progress_rate"] == 100.0
