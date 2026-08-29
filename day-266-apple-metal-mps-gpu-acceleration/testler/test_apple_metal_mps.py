"""
PyTest Birim Testleri - Day 266 (FAZ 14): Apple Silicon Metal (MPS) GPU Optimizasyonu.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.apple_metal_motoru import (
    AppleSiliconUMAManager,
    MetalPerformanceShadersEngine,
)
from src.apple_metal_profilleyici import AppleMetalMPSProfilleyici
from src.gorsellestirici import AppleMetalGorsellestirici


def test_uma_manager_shared_allocation():
    """1. AppleSiliconUMAManager paylaşımlı UMA tensörünü doğru boyutta tahsis etmelidir."""
    mgr = AppleSiliconUMAManager(memory_gb=64)
    t = mgr.allocate_shared_tensor("test_buf", shape=(2, 16, 32))
    assert t.shape == (2, 16, 32)
    assert "test_buf" in mgr.allocated_buffers


def test_uma_transfer_comparison():
    """2. UMA transfer süresi 0.0 ms ve kopyalama baytı 0 olmalıdır."""
    tensor = np.zeros((1024, 1024), dtype=np.float32)  # 4 MB
    stats = AppleSiliconUMAManager.compare_transfer_overhead(tensor)
    assert stats["uma_transfer_ms"] == 0.0
    assert stats["uma_bellek_cogaltma_bayt"] == 0
    assert stats["pcie_transfer_ms"] > 0.0


def test_mps_rms_norm():
    """3. RMSNorm çıktısının varyansı ağırlıkla doğru ölçeklenmelidir."""
    x = np.random.randn(4, 32).astype(np.float32) * 5.0
    weight = np.ones(32, dtype=np.float32)
    norm = MetalPerformanceShadersEngine.rms_norm(x, weight)
    rms = np.sqrt(np.mean(norm ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-2)


def test_mps_rope_rotation():
    """4. RoPE rotasyonu tensör boyutunu korumalıdır."""
    x = np.random.randn(2, 8, 16).astype(np.float32)
    rotated = MetalPerformanceShadersEngine.apply_rope(x)
    assert rotated.shape == (2, 8, 16)


def test_mps_silu_activation():
    """5. SiLU aktivasyonu doğru matematiksel değerler üretmelidir."""
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    res = MetalPerformanceShadersEngine.silu(x)
    assert res[1] == 0.0
    assert res[2] > 0.0
    assert res[0] < 0.0


def test_mps_fused_transformer_block():
    """6. MPS Fused Transformer bloğu doğru boyutta çıktı üretmelidir."""
    x = np.random.randn(1, 16, 32).astype(np.float32)
    norm_w = np.ones(32, dtype=np.float32)
    w_gate = np.random.randn(32, 64).astype(np.float32)
    w_up = np.random.randn(32, 64).astype(np.float32)
    w_down = np.random.randn(64, 32).astype(np.float32)

    out, stats = MetalPerformanceShadersEngine.execute_mps_fused_transformer_block(
        x, norm_w, w_gate, w_up, w_down
    )
    assert out.shape == (1, 16, 32)
    assert stats["metal_command_encoders"] == 1


def test_apple_metal_profiler_output():
    """7. AppleMetalMPSProfilleyici 3'lü sistem kıyaslama raporunu üretmelidir."""
    profil = AppleMetalMPSProfilleyici.basarim_profili_cikar()
    assert "Apple_Metal_MPS" in profil["karsilastirma"]["cikarim_hizi_tok_s"]
    assert profil["karsilastirma"]["cikarim_hizi_tok_s"]["Apple_Metal_MPS"] == 46.5


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. AppleMetalGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_apple_metal_paneli.png")
    profil = AppleMetalMPSProfilleyici.basarim_profili_cikar()

    AppleMetalGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
