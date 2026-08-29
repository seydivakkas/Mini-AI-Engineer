"""
PyTest Birim Testleri - Day 265 (FAZ 14): Triton Fused MoE Expert Routing.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fused_moe_motoru import NaiveMoERouter, TritonFusedMoERouter
from src.fused_moe_profilleyici import FusedMoEProfilleyici
from src.gorsellestirici import FusedMoEGorsellestirici


def test_naive_moe_output_shape():
    """1. NaiveMoERouter doğru tensör boyutunu üretmelidir."""
    x = np.random.randn(64, 32).astype(np.float32)
    w_gate = np.random.randn(32, 4).astype(np.float32)
    expert_weights = [np.random.randn(32, 32).astype(np.float32) for _ in range(4)]

    out, stats = NaiveMoERouter.forward(x, w_gate, expert_weights, top_k=2)
    assert out.shape == (64, 32)
    assert stats["toplam_kopyalanan_bayt"] > 0


def test_fused_moe_output_shape():
    """2. TritonFusedMoERouter doğru tensör boyutunu üretmelidir."""
    x = np.random.randn(64, 32).astype(np.float32)
    w_gate = np.random.randn(32, 4).astype(np.float32)
    expert_weights = [np.random.randn(32, 32).astype(np.float32) for _ in range(4)]

    out, stats = TritonFusedMoERouter.forward(x, w_gate, expert_weights, top_k=2)
    assert out.shape == (64, 32)
    assert stats["toplam_kopyalanan_bayt"] == 0


def test_fused_vs_naive_mathematical_identity():
    """3. Triton Fused MoE çıktısı Naive MoE ile matematiksel olarak özdeş olmalıdır."""
    np.random.seed(42)
    x = np.random.randn(64, 32).astype(np.float32)
    w_gate = np.random.randn(32, 4).astype(np.float32)
    expert_weights = [np.random.randn(32, 32).astype(np.float32) for _ in range(4)]

    out_naive, _ = NaiveMoERouter.forward(x, w_gate, expert_weights, top_k=2)
    out_fused, _ = TritonFusedMoERouter.forward(x, w_gate, expert_weights, top_k=2)

    assert np.allclose(out_naive, out_fused, atol=1e-4)


def test_fused_moe_zero_copy_verification():
    """4. Triton Fused MoE bellek kopyalama baytını 0 olarak raporlamalıdır."""
    x = np.random.randn(32, 16).astype(np.float32)
    w_gate = np.random.randn(16, 4).astype(np.float32)
    expert_weights = [np.random.randn(16, 16).astype(np.float32) for _ in range(4)]

    _, stats = TritonFusedMoERouter.forward(x, w_gate, expert_weights, top_k=1)
    assert stats["toplam_kopyalanan_bayt"] == 0


def test_fused_moe_topk_variation():
    """5. Farklı Top-k değerlerinde (k=1, k=3) Fused ve Naive aynı sonucu vermelidir."""
    np.random.seed(123)
    x = np.random.randn(32, 16).astype(np.float32)
    w_gate = np.random.randn(16, 6).astype(np.float32)
    expert_weights = [np.random.randn(16, 16).astype(np.float32) for _ in range(6)]

    out_n1, _ = NaiveMoERouter.forward(x, w_gate, expert_weights, top_k=1)
    out_f1, _ = TritonFusedMoERouter.forward(x, w_gate, expert_weights, top_k=1)
    assert np.allclose(out_n1, out_f1, atol=1e-4)

    out_n3, _ = NaiveMoERouter.forward(x, w_gate, expert_weights, top_k=3)
    out_f3, _ = TritonFusedMoERouter.forward(x, w_gate, expert_weights, top_k=3)
    assert np.allclose(out_n3, out_f3, atol=1e-4)


def test_fused_moe_large_expert_count():
    """6. 16 uzmanlı MoE mimarisinde hatasız yönlendirme yapılmalıdır."""
    x = np.random.randn(64, 16).astype(np.float32)
    w_gate = np.random.randn(16, 16).astype(np.float32)
    expert_weights = [np.random.randn(16, 16).astype(np.float32) for _ in range(16)]

    out_fused, stats = TritonFusedMoERouter.forward(x, w_gate, expert_weights, top_k=2)
    assert out_fused.shape == (64, 16)
    assert stats["uzman_sayisi"] == 16


def test_fused_moe_profiler_output():
    """7. FusedMoEProfilleyici 3'lü karşılaştırma raporunu üretmelidir."""
    profil = FusedMoEProfilleyici.basarim_profili_cikar()
    assert "Triton_Fused_MoE" in profil["karsilastirma"]["uctan_uca_gecikme_ms"]
    assert profil["karsilastirma"]["uctan_uca_gecikme_ms"]["Triton_Fused_MoE"] == 3.9


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. FusedMoEGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_fused_moe_paneli.png")
    profil = FusedMoEProfilleyici.basarim_profili_cikar()

    FusedMoEGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
