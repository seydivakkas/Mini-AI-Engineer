"""
PyTest Birim Testleri - Day 296 (FAZ 15): Otonom Donanım Tasarımı ve HLS/Verilog Sentezi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hardware_synthesis_motoru import (
    HardwareSpec,
    HLSOptimizer,
    VerilogRTLGenerator,
    FPGATimingAnalyzer,
)
from src.hardware_synthesis_profilleyici import HardwareSynthesisProfilleyici
from src.gorsellestirici import HardwareSynthesisGorsellestirici


def test_hardware_spec_initialization():
    """1. Donanım özellikleri matris boyutu, hassasiyet ve PE sayısıyla başlatılmalıdır."""
    spec = HardwareSpec(array_size=16, precision="INT8", target_clock_mhz=500.0)
    assert spec.array_size == 16
    assert spec.total_pes == 256
    assert spec.precision == "INT8"


def test_hls_optimizer_pipeline_ii():
    """2. HLS optimize edici II=1 boru hattı ve DSP tahsisi üretmelidir."""
    spec = HardwareSpec(array_size=16)
    opt = HLSOptimizer.optimize_spec(spec)
    assert opt["pipeline_ii"] == 1
    assert opt["dsp_blocks"] == 256


def test_verilog_rtl_generator_content():
    """3. SystemVerilog üreteci sentezlenebilir modül ve sistolik dizi kodları üretmelidir."""
    spec = HardwareSpec(array_size=8)
    rtl = VerilogRTLGenerator.generate_systemverilog(spec)
    assert "module systolic_array_top" in rtl
    assert "input  wire                   clk" in rtl
    assert "pe_int8 pe_inst" in rtl


def test_fpga_timing_analyzer_closure():
    """4. FPGA zamanlama analizörü pozitif WNS (>0.0ns) ve kapanış bildirmelidir."""
    spec = HardwareSpec(target_clock_mhz=500.0)
    timing = FPGATimingAnalyzer.analyze_timing(spec)
    assert timing["wns_ns"] > 0.0
    assert timing["timing_met"] is True
    assert timing["achieved_fmax_mhz"] >= 500.0


def test_profiler_synthesis_speedup():
    """5. Donanım tasarım süresi hızlanma oranı 10,000 kattan fazla olmalıdır."""
    profil = HardwareSynthesisProfilleyici.basarim_profili_cikar()
    assert profil["hizlanma_orani"] >= 10000.0


def test_profiler_energy_efficiency_superiority():
    """6. Enerji verimliliği 15.0 TFLOPS/W'ın üzerinde olmalıdır."""
    profil = HardwareSynthesisProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["enerji_verimliligi_tflops_w"]["3. AI Hardware Engine"] > 15.0


def test_profiler_clock_frequency_target():
    """7. Saat frekansı hedefi (500 MHz) karşılanmış olmalıdır."""
    profil = HardwareSynthesisProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["saat_frekansi_mhz"]["3. AI Hardware Engine"] >= 500.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. HardwareSynthesisGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_hardware_paneli.png")
    profil = HardwareSynthesisProfilleyici.basarim_profili_cikar()

    HardwareSynthesisGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
