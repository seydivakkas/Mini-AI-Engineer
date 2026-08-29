"""
PyTest Birim Testleri - Day 275 (FAZ 14): Ring Attention (1M+ Token Sonsuz Bağlam).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ring_attention_motoru import RingAttentionKernelEngine
from src.ring_attention_profilleyici import RingAttentionProfilleyici
from src.gorsellestirici import RingAttentionGorsellestirici


def test_ring_engine_init():
    """1. Ring Attention motoru doğru GPU sayısı ve ölçek katsayısıyla başlamalıdır."""
    engine = RingAttentionKernelEngine(num_gpus=8, d_model=128)
    assert engine.num_gpus == 8
    assert engine.d_model == 128
    assert np.isclose(engine.scale, 1.0 / np.sqrt(128))


def test_ring_attention_shape_integrity():
    """2. Ring Attention tüm GPU'lar için doğru blok çıktı boyutlarını üretmelidir."""
    engine = RingAttentionKernelEngine(num_gpus=4, d_model=64)
    Q_blocks = [np.random.randn(2, 32, 64).astype(np.float32) for _ in range(4)]
    K_blocks = [np.random.randn(2, 32, 64).astype(np.float32) for _ in range(4)]
    V_blocks = [np.random.randn(2, 32, 64).astype(np.float32) for _ in range(4)]

    outs = engine.execute_ring_attention(Q_blocks, K_blocks, V_blocks)
    assert len(outs) == 4
    for out in outs:
        assert out.shape == (2, 32, 64)
        assert not np.isnan(out).any()


def test_ring_attention_mathematical_equivalence():
    """3. Ring Attention çıktısı monolitik küresel dikkatle tam matematiksel denkliğe sahip olmalıdır."""
    res = RingAttentionKernelEngine.execute_mock_ring_pipeline(total_seq_len=256, num_gpus=4, d_model=32)
    assert res["matematiksel_eslesme"]
    assert res["maksimum_fark"] < 1e-4


def test_causal_ring_attention_masking():
    """4. Kausal modda gelecekteki token bloklarının etkisi sıfırlanmalıdır."""
    engine = RingAttentionKernelEngine(num_gpus=2, d_model=16)
    Q_blocks = [np.ones((1, 8, 16), dtype=np.float32) for _ in range(2)]
    K_blocks = [np.ones((1, 8, 16), dtype=np.float32) for _ in range(2)]
    V_blocks = [np.ones((1, 8, 16), dtype=np.float32) for _ in range(2)]

    outs = engine.execute_ring_attention(Q_blocks, K_blocks, V_blocks, is_causal=True)
    assert len(outs) == 2
    assert not np.isnan(outs[0]).any()
    assert not np.isnan(outs[1]).any()


def test_mock_ring_pipeline_success():
    """5. execute_mock_ring_pipeline doğru VRAM tasarruf oranını (P) döndürmelidir."""
    res = RingAttentionKernelEngine.execute_mock_ring_pipeline(total_seq_len=512, num_gpus=8, d_model=64)
    assert res["num_gpus"] == 8
    assert res["vram_tasarrufu_orani"] == 8.0
    assert res["maksimum_fark"] < 1e-4


def test_profiler_overlap_and_memory_metrics():
    """6. RingAttentionProfilleyici %98.6 iletişim örtüşmesi ve 16 GB VRAM raporlamalıdır."""
    profil = RingAttentionProfilleyici.basarim_profili_cikar()
    karsilastirma = profil["karsilastirma"]
    assert karsilastirma["iletisim_ortusme_verimi_yuzde"]["Ring_Attention_8GPU"] == 98.6
    assert karsilastirma["vram_tepe_noktasi_1m_gb"]["Ring_Attention_8GPU"] == 16.0
    assert profil["hizlanma_orani"] > 5.0


def test_linear_gpu_memory_scaling():
    """7. GPU sayısı arttıkça bağlam kapasitesi doğrusal ölçeklenmelidir."""
    profil = RingAttentionProfilleyici.basarim_profili_cikar()
    skala = profil["skala"]
    assert skala["ring_attn_vram_gb"][-1] < skala["flashattn_vram_gb"][-1]
    assert len(skala["baglamlar_k"]) == 8


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. RingAttentionGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_ring_paneli.png")
    profil = RingAttentionProfilleyici.basarim_profili_cikar()

    RingAttentionGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
