"""
PyTest Birim Testleri - Day 280 (FAZ 14): Ultra-Low-Bit Hardware Grand Capstone.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grand_capstone_motoru import HardwareGrandCapstoneEngine
from src.grand_capstone_profilleyici import GrandCapstoneProfilleyici
from src.gorsellestirici import GrandCapstoneGorsellestirici


def test_pack_ternary_weights_shapes_and_codes():
    """1. Ternary ağırlık paketleme K boyutunu 16'ya bölerek UINT32 matris üretmelidir."""
    w = np.random.randn(64, 32).astype(np.float32)
    w_packed, gamma = HardwareGrandCapstoneEngine.pack_ternary_weights(w)
    
    assert w_packed.shape == (4, 32)
    assert w_packed.dtype == np.uint32
    assert gamma > 0.0


def test_fused_bitlinear_fp8_gemm_execution():
    """2. Fused BitLinear GEMM doğru çıktı boyutunu ve sonlu değerler üretmelidir."""
    x = np.random.randn(4, 64).astype(np.float32)
    w = np.random.randn(64, 32).astype(np.float32)
    w_packed, gamma = HardwareGrandCapstoneEngine.pack_ternary_weights(w)
    
    out = HardwareGrandCapstoneEngine.fused_bitlinear_fp8_gemm(x, w_packed, gamma, out_dim=32)
    assert out.shape == (4, 32)
    assert not np.isnan(out).any()


def test_flash_decoding_step_numerical_equivalence():
    """3. FlashDecoding++ Split-KV dikkat adımı monolitik dikkatle birebir denk olmalıdır."""
    q = np.random.randn(2, 1, 64).astype(np.float32)
    k = np.random.randn(2, 256, 64).astype(np.float32)
    v = np.random.randn(2, 256, 64).astype(np.float32)
    
    out_split = HardwareGrandCapstoneEngine.flash_decoding_step(q, k, v, num_splits=4)
    
    scale = 1.0 / np.sqrt(64)
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    out_ref = np.matmul(weights, v)
    
    fark = np.max(np.abs(out_ref - out_split))
    assert fark < 1e-4


def test_execute_grand_capstone_layer_pipeline():
    """4. execute_grand_capstone_layer uçtan uca katmanı başarıyla doğrulamalıdır."""
    x = np.random.randn(2, 16, 64).astype(np.float32)
    w = np.random.randn(64, 64).astype(np.float32)
    k = np.random.randn(2, 128, 64).astype(np.float32)
    v = np.random.randn(2, 128, 64).astype(np.float32)
    
    res = HardwareGrandCapstoneEngine.execute_grand_capstone_layer(x, w, k, v)
    assert res["matematiksel_dogruluk"] is True
    assert res["attn_error"] < 1e-4
    assert res["vram_sikistirma_orani"] == 8.2


def test_profiler_vram_and_energy_savings():
    """5. GrandCapstoneProfilleyici 8.1x VRAM ve 4.6x enerji tasarrufu raporlamalıdır."""
    profil = GrandCapstoneProfilleyici.basarim_profili_cikar()
    assert profil["vram_kazanci"] > 8.0
    assert profil["enerji_kazanci"] > 4.5


def test_profiler_throughput_and_mfu_metrics():
    """6. GrandCapstoneProfilleyici 154 tok/s ve %74.5 MFU raporlamalıdır."""
    profil = GrandCapstoneProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["token_throughput_tok_s"]["FAZ14_Grand_Capstone"] == 154.0
    assert kars["model_flops_utilization_mfu"]["FAZ14_Grand_Capstone"] == 74.5
    assert profil["hizlanma_orani"] > 8.0


def test_flash_decoding_split_scaling():
    """7. Farklı split boyutlarında (2, 4, 8) FlashDecoding++ stabil çalışmalıdır."""
    q = np.random.randn(1, 1, 32).astype(np.float32)
    k = np.random.randn(1, 64, 32).astype(np.float32)
    v = np.random.randn(1, 64, 32).astype(np.float32)
    
    out2 = HardwareGrandCapstoneEngine.flash_decoding_step(q, k, v, num_splits=2)
    out8 = HardwareGrandCapstoneEngine.flash_decoding_step(q, k, v, num_splits=8)
    assert np.allclose(out2, out8, atol=1e-4)


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. GrandCapstoneGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_capstone_paneli.png")
    profil = GrandCapstoneProfilleyici.basarim_profili_cikar()

    GrandCapstoneGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
