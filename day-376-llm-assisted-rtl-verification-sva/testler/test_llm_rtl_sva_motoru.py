"""
Day 376: Unit Tests for LLM-Assisted RTL Verification and SVA Generation
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from llm_rtl_sva_motoru import (
    LLMSVASynthesizer,
    SVAProperty,
    RTLTraceEvaluator,
    LLMRTLSVABenchmark
)


def test_sva_sentezleme_ve_sozdizimi():
    """LLM SVA sentezleyicisinin geçerli SystemVerilog iddiaları üretmesini test eder."""
    synthesizer = LLMSVASynthesizer()
    spec = "FIFO asla doluyken yazma yapmamalı ve boşken okumamalıdır."
    assertions = synthesizer.sentezle_sva(spec)

    assert len(assertions) >= 3, "En az 3 SVA iddiası sentezlenmelidir."
    for sva in assertions:
        assert isinstance(sva, SVAProperty)
        assert "assert property" in sva.sva_code
        assert sva.name.startswith("sva_")


def test_fifo_overflow_underflow_ihlal_yakalama():
    """FIFO taşma ve boşalma hatalarının SVA tarafından yakalandığını test eder."""
    synthesizer = LLMSVASynthesizer()
    assertions = synthesizer.sentezle_sva("FIFO şartnamesi")
    evaluator = RTLTraceEvaluator(assertions)

    # 1. Taşma İhlali Enjeksiyonu
    bad_overflow_trace = {
        "clk": [1, 1],
        "count": [8, 8],
        "wr": [1, 1],
        "rd": [0, 0],
        "valid": [0, 0],
        "ready": [0, 0],
        "data": [0, 0],
        "grant_bus": [0, 0]
    }
    evaluator.degerlendir_trace(bad_overflow_trace, inject_overflow_bug=True)
    overflow_sva = next(a for a in assertions if a.name == "sva_fifo_no_overflow")
    assert overflow_sva.failed_count > 0, "Taşma hatası SVA ihlali tetiklemelidir!"

    # 2. Boşalma İhlali Enjeksiyonu
    bad_underflow_trace = {
        "clk": [1, 1],
        "count": [0, 0],
        "wr": [0, 0],
        "rd": [1, 1],
        "valid": [0, 0],
        "ready": [0, 0],
        "data": [0, 0],
        "grant_bus": [0, 0]
    }
    evaluator.degerlendir_trace(bad_underflow_trace, inject_underflow_bug=True)
    underflow_sva = next(a for a in assertions if a.name == "sva_fifo_no_underflow")
    assert underflow_sva.failed_count > 0, "Boşalma hatası SVA ihlali tetiklemelidir!"


def test_axi_handshake_kararlilik_dogrulama():
    """AXI-Stream valid/ready el sıkışmasında veri kararsızlığı hatasının yakalandığını test eder."""
    synthesizer = LLMSVASynthesizer()
    assertions = synthesizer.sentezle_sva("AXI handshake")
    evaluator = RTLTraceEvaluator(assertions)

    # Valid çekilmiş ama ready=0 iken veri değişirse ihlal olmalı
    trace = {
        "clk": [1, 1],
        "count": [2, 2],
        "wr": [0, 0],
        "rd": [0, 0],
        "valid": [1, 1],
        "ready": [0, 0],
        "data": [42, 99],  # Veri mutate edildi!
        "grant_bus": [1, 1]
    }
    evaluator.degerlendir_trace(trace, inject_axi_bug=True)
    axi_sva = next(a for a in assertions if a.name == "sva_axi_handshake_stability")
    assert axi_sva.failed_count > 0, "AXI el sıkışma kararsızlığı SVA ihlali tetiklemelidir!"


def test_tam_benchmark_ve_hizlanma_orani():
    """Tam benchmark akışını, %100 hata tespitini ve 8.5x hızlanma faktörünü test eder."""
    bench = LLMRTLSVABenchmark()
    res = bench.kos(num_cycles=200)

    assert res["bugs_injected"] > 0
    assert res["bugs_detected"] == res["bugs_injected"], "Tüm enjekte edilen köşe hatalar yakalanmalıdır!"
    assert res["detection_rate"] == 100.0, "Hata tespit oranı %100 olmalıdır."
    assert res["speedup_x"] >= 8.0, "LLM SVA doğrulaması en az 8x hızlanma sağlamalıdır."
