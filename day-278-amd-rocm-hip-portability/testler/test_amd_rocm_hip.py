"""
PyTest Birim Testleri - Day 278 (FAZ 14): AMD ROCm & HIP Taşınabilirliği.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hip_donusturucu_motoru import HIPPortabilityEngine
from src.hip_profilleyici import HIPProfilleyici
from src.gorsellestirici import HIPGorsellestirici


def test_transpile_cuda_to_hip_syntax():
    """1. CUDA API çağrıları HIP karşılıklarına eksiksiz dönüştürülmelidir."""
    cuda_code = "cudaMalloc(&p, 100); cudaMemcpy(p, q, 100, cudaMemcpyHostToDevice);"
    res = HIPPortabilityEngine.transpile_cuda_to_hip(cuda_code)
    
    assert "hipMalloc" in res["hip_kodu"]
    assert "hipMemcpy" in res["hip_kodu"]
    assert "hipMemcpyHostToDevice" in res["hip_kodu"]
    assert res["toplam_donusum"] >= 3


def test_transpile_includes_hip_header():
    """2. Dönüştürülen HIP kodunda hip_runtime başlığı yer almalıdır."""
    res = HIPPortabilityEngine.transpile_cuda_to_hip("int x = 5;")
    assert "#include <hip/hip_runtime.h>" in res["hip_kodu"]


def test_cdna3_mfma_gemm_shape_and_accuracy():
    """3. AMD CDNA3 MFMA simülasyonu referans matris çarpımıyla birebir eşleşmelidir."""
    np.random.seed(42)
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)
    
    out, stats = HIPPortabilityEngine.execute_cdna3_mfma_gemm(a, b)
    assert out.shape == (32, 32)
    assert stats["matematiksel_eslesme"] is True
    assert stats["maksimum_fark"] < 1e-4


def test_wavefront_64_alignment():
    """4. AMD CDNA3 ortamında Wavefront boyutu 64 thread olarak raporlanmalıdır."""
    a = np.random.randn(16, 16).astype(np.float32)
    b = np.random.randn(16, 16).astype(np.float32)
    _, stats = HIPPortabilityEngine.execute_cdna3_mfma_gemm(a, b)
    assert stats["wavefront_size"] == 64
    assert "__builtin_amdgcn_mfma" in stats["kullanilan_mfma_instruction"]


def test_h100_vs_mi300x_spec_advantages():
    """5. MI300X VRAM (192 GB) ve bant genişliği (5.3 TB/s) avantajları doğru olmalıdır."""
    profil = HIPProfilleyici.basarim_profili_cikar()
    assert profil["vram_avantaj_orani"] == 2.4
    assert np.isclose(profil["bant_avantaj_orani"], 5.30 / 3.35, atol=0.01)


def test_hip_profilleyici_reports_and_concurrency():
    """6. HIPProfilleyici MI300X üzerinde Batch 64 ve 210 tok/s raporlamalıdır."""
    profil = HIPProfilleyici.basarim_profili_cikar()
    karsilastirma = profil["karsilastirma"]
    assert karsilastirma["llama_70b_tek_gpu_maks_batch"]["AMD_Instinct_MI300X"] == 64.0
    assert karsilastirma["llama_70b_token_throughput_tok_s"]["AMD_Instinct_MI300X"] == 210.0


def test_mfma_pipeline_efficiency_values():
    """7. MFMA donanım boru hattı verimliliği tüm aşamalarda %95 üzerinde olmalıdır."""
    profil = HIPProfilleyici.basarim_profili_cikar()
    verimler = profil["mfma_asamalari"]["verimlilik_yuzde"]
    for v in verimler:
        assert v > 95.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. HIPGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_hip_paneli.png")
    profil = HIPProfilleyici.basarim_profili_cikar()

    HIPGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
