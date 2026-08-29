"""
PyTest Birim Testleri - Day 268 (FAZ 14): Apache TVM & IREE Edge NPU Derleme Optimizasyonu.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tvm_edge_npu_motoru import (
    TVMTensorIRCompilerEngine,
    HexagonEthosNPUCodeGen,
)
from src.tvm_edge_npu_profilleyici import TVMEdgeNPUProfilleyici
from src.gorsellestirici import TVMEdgeNPUGorsellestirici


def test_tvm_gelu_activation():
    """1. GELU aktivasyonu doğru matematiksel değerler üretmelidir."""
    x = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
    res = TVMTensorIRCompilerEngine.gelu(x)
    assert res[1] == 0.0
    assert res[2] > 2.9
    assert abs(res[0]) < 0.01


def test_tvm_fused_gemm_shape():
    """2. TensorIR Fused GEMM doğru çıktı boyutunu üretmelidir."""
    x = np.random.randn(32, 64).astype(np.float32)
    w = np.random.randn(64, 48).astype(np.float32)
    bias = np.random.randn(48).astype(np.float32)

    out, stats = TVMTensorIRCompilerEngine.execute_fused_gemm_bias_gelu(x, w, bias, tile_size=16)
    assert out.shape == (32, 48)
    assert stats["ara_dram_yazma_bayt"] == 0


def test_tvm_fused_gemm_mathematical_identity():
    """3. TensorIR Fused çıktısı klasik ayrık hesaplama ile özdeş olmalıdır."""
    np.random.seed(42)
    x = np.random.randn(32, 32).astype(np.float32)
    w = np.random.randn(32, 32).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)

    out_fused, _ = TVMTensorIRCompilerEngine.execute_fused_gemm_bias_gelu(x, w, bias, tile_size=16)
    ref = TVMTensorIRCompilerEngine.gelu(np.dot(x, w) + bias)

    assert np.allclose(out_fused, ref, atol=1e-4)


def test_tvm_fused_gemm_arbitrary_sizes():
    """4. 16'nın katı olmayan boyutlarda da (ör. 23x37 ve 37x29) hatasız çalışmalıdır."""
    np.random.seed(123)
    x = np.random.randn(23, 37).astype(np.float32)
    w = np.random.randn(37, 29).astype(np.float32)
    bias = np.random.randn(29).astype(np.float32)

    out_fused, _ = TVMTensorIRCompilerEngine.execute_fused_gemm_bias_gelu(x, w, bias, tile_size=16)
    ref = TVMTensorIRCompilerEngine.gelu(np.dot(x, w) + bias)

    assert np.allclose(out_fused, ref, atol=1e-4)


def test_tvm_fused_gemm_zero_dram_writes():
    """5. Fused operatör ara tensör DRAM yazmasını 0 bayt olarak bildirmelidir."""
    x = np.random.randn(16, 16).astype(np.float32)
    w = np.random.randn(16, 16).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)

    _, stats = TVMTensorIRCompilerEngine.execute_fused_gemm_bias_gelu(x, w, bias, tile_size=16)
    assert stats["ara_dram_yazma_bayt"] == 0


def test_hexagon_ethos_codegen_c_source():
    """6. NPU C kod üreticisi geçerli fonksiyon imzası ve unroll direktifleri üretmelidir."""
    res = HexagonEthosNPUCodeGen.generate_standalone_c_source("test_layer", 32, 32, 32)
    code = res["c_kaynak_kodu"]
    assert "void test_layer_fused_npu_kernel" in code
    assert "#pragma unroll" in code
    assert "__restrict__" in code
    assert res["ikili_boyut_kb"] < 500


def test_tvm_edge_npu_profiler_output():
    """7. TVMEdgeNPUProfilleyici 3'lü karşılaştırma raporunu üretmelidir."""
    profil = TVMEdgeNPUProfilleyici.basarim_profili_cikar()
    assert "TVM_Fused_NPU" in profil["karsilastirma"]["cikarim_gecikmesi_ms"]
    assert profil["karsilastirma"]["cikarim_gecikmesi_ms"]["TVM_Fused_NPU"] == 2.8


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. TVMEdgeNPUGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_tvm_edge_paneli.png")
    profil = TVMEdgeNPUProfilleyici.basarim_profili_cikar()

    TVMEdgeNPUGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
