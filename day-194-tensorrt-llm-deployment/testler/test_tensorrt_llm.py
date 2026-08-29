"""
PyTest Birim Testleri - Day 194: TensorRT-LLM Derleme ve FP8 Tensor Core.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tensorrt_llm_motoru import (
    FP8QuantizationSimulator,
    TRTLLMEngineCompiler,
    InFlightBatchingRuntime,
)
from src.trt_profilleyici import TRTLLMBenchmarkProfilleyici
from src.gorsellestirici import TRTLLMGorsellestirici


def test_fp8_kuantizasyon_araligi():
    """1. FP8 kuantizasyonu [-448.0, 448.0] aralığını aşmamalıdır."""
    x = torch.randn(10, 10) * 100.0
    x_deq, scale = FP8QuantizationSimulator.kuantize_et(x)
    assert scale > 0.0
    assert torch.allclose(x, x_deq, atol=2.0)


def test_fp8_gemm_cikti_sekli():
    """2. FP8 GEMM matris çarpım şeklini doğru üretmelidir."""
    a = torch.randn(4, 64)
    b = torch.randn(64, 128)
    out = FP8QuantizationSimulator.gemm_fp8(a, b)
    assert out.shape == (4, 128)


def test_trt_llm_derleyici_raporu():
    """3. TRTLLMEngineCompiler 5 adımlı optimizasyon raporunu üretmelidir."""
    compiler = TRTLLMEngineCompiler(hidden_dim=128, intermediate_dim=256, use_fp8=True)
    rapor = compiler.compile()
    assert rapor["durum"] == "DERLENDİ (COMPILED)"
    assert len(rapor["derleme_adimlari"]) == 5


def test_in_flight_batching_runtime_ileri_gecis():
    """4. In-Flight Batching çalışma zamanı girdi boyutunu koruyarak ileri geçişi tamamlamalıdır."""
    compiler = TRTLLMEngineCompiler(hidden_dim=64, intermediate_dim=128, use_fp8=True)
    compiler.compile()
    runtime = InFlightBatchingRuntime(engine=compiler)

    x = torch.randn(2, 64)
    out = runtime.forward_step(x)
    assert out.shape == (2, 64)


def test_fp8_olmadan_derleme_ve_calistirma():
    """5. use_fp8=False durumunda standart FP16 GEMM ile hatasız çalışmalıdır."""
    compiler = TRTLLMEngineCompiler(hidden_dim=32, intermediate_dim=64, use_fp8=False)
    compiler.compile()
    runtime = InFlightBatchingRuntime(engine=compiler)

    x = torch.randn(1, 32)
    out = runtime.forward_step(x)
    assert out.shape == (1, 32)


def test_trt_profilleyici_tam_model_raporu():
    """6. Kıyaslama raporu 3 motoru ve 4.0x tepe hızlanmayı içermelidir."""
    rapor = TRTLLMBenchmarkProfilleyici.tam_model_kiyaslama_raporu()
    assert len(rapor) == 3
    trt_motor = rapor[2]
    assert "TensorRT-LLM" in trt_motor["motor_adi"]
    assert trt_motor["model_vram_gb"] == 70.0


def test_trt_profilleyici_batch_olcekleme():
    """7. Batch ölçekleme raporu batch 128'de yüksek throughput doğrulamalıdır."""
    rapor = TRTLLMBenchmarkProfilleyici.batch_olcekleme_analizi()
    assert len(rapor) == 5
    son = rapor[-1]
    assert son["batch_size"] == 128
    assert son["trt_llm_tokens_sec"] > son["pytorch_tokens_sec"]


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. TRTLLMGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_trt_paneli.png")
    kiyas_raporu = TRTLLMBenchmarkProfilleyici.tam_model_kiyaslama_raporu()
    batch_raporu = TRTLLMBenchmarkProfilleyici.batch_olcekleme_analizi()

    TRTLLMGorsellestirici.teshis_paneli_olustur(
        kiyas_raporu=kiyas_raporu,
        batch_raporu=batch_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
