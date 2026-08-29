"""
Grouped-Query Attention (GQA) Birim ve Entegrasyon Testleri (Day 102).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.kv_cache import KVCache
from src.dikkat_mimarileri import AttentionTuru, repeat_kv, GroupedQueryAttention
from src.karsilastirma_motoru import GQALaboratuvari
from src.gorsellestirici import GQAGorsellestirici


def test_kv_cache_guncelleme_ve_boyut():
    """KVCache tensör ekleme, indeksleme ve sıfırlama işlemlerini test eder."""
    cache = KVCache(max_batch_size=4, max_seq_len=64, num_kv_heads=2, head_dim=16)
    k1 = torch.randn(2, 2, 8, 16)
    v1 = torch.randn(2, 2, 8, 16)

    aktif_k, aktif_v = cache.guncelle(k1, v1)
    assert aktif_k.shape == (2, 2, 8, 16)
    assert cache.mevcut_uzunluk == 8

    # 1 token daha ekle
    k2 = torch.randn(2, 2, 1, 16)
    v2 = torch.randn(2, 2, 1, 16)
    aktif_k2, aktif_v2 = cache.guncelle(k2, v2)
    assert aktif_k2.shape == (2, 2, 9, 16)
    assert cache.mevcut_uzunluk == 9

    cache.sifirla()
    assert cache.mevcut_uzunluk == 0


def test_kv_cache_bellek_hesabi():
    """KVCache teorik bellek hesaplama fonksiyonunu doğrular."""
    cache = KVCache(max_batch_size=1, max_seq_len=1024, num_kv_heads=8, head_dim=64, dtype=torch.float16)
    # 2 * 32 * 1 * 8 * 1024 * 64 * 2 = 67,108,864 bayt = 64 MB
    mb = cache.bellek_tuketimi_mb(katman_sayisi=32)
    assert round(mb, 1) == 64.0


def test_repeat_kv_fonksiyonu():
    """repeat_kv fonksiyonunun KV başlıklarını doğru oranda çoğalttığını test eder."""
    # 2 KV başlığını 4 kat çoğaltarak 8 başlığa eşle
    x = torch.randn(2, 2, 10, 16)
    repeated = repeat_kv(x, n_rep=4)
    assert repeated.shape == (2, 8, 10, 16)
    # İlk 4 başlığın orijinal 1. başlıkla aynı olduğunu doğrula
    for i in range(4):
        assert torch.allclose(repeated[:, i, :, :], x[:, 0, :, :])


def test_mha_ileri_gecis():
    """Multi-Head Attention (num_q_heads == num_kv_heads) ileri geçişini test eder."""
    mha = GroupedQueryAttention(dim=64, num_q_heads=4, num_kv_heads=4)
    assert mha.mimari_turu == AttentionTuru.MHA
    x = torch.randn(2, 16, 64)
    out, _ = mha(x)
    assert out.shape == (2, 16, 64)


def test_mqa_ileri_gecis():
    """Multi-Query Attention (num_kv_heads == 1) ileri geçişini test eder."""
    mqa = GroupedQueryAttention(dim=64, num_q_heads=4, num_kv_heads=1)
    assert mqa.mimari_turu == AttentionTuru.MQA
    x = torch.randn(2, 16, 64)
    out, _ = mqa(x)
    assert out.shape == (2, 16, 64)


def test_gqa_ileri_gecis_ve_kv_cache():
    """Grouped-Query Attention (GQA) katmanının KVCache ile adım adım çalışmasını test eder."""
    gqa = GroupedQueryAttention(dim=64, num_q_heads=4, num_kv_heads=2)
    assert gqa.mimari_turu == AttentionTuru.GQA
    cache = KVCache(max_batch_size=2, max_seq_len=32, num_kv_heads=2, head_dim=16)

    # 1. Prefill Aşaması (8 token)
    x_prompt = torch.randn(2, 8, 64)
    out1, cache = gqa(x_prompt, kv_cache=cache, is_causal=True)
    assert out1.shape == (2, 8, 64)
    assert cache.mevcut_uzunluk == 8

    # 2. Decode Aşaması (1 token)
    x_token = torch.randn(2, 1, 64)
    out2, cache = gqa(x_token, kv_cache=cache, is_causal=False)
    assert out2.shape == (2, 1, 64)
    assert cache.mevcut_uzunluk == 9


def test_gqa_laboratuvari_karsilastirma():
    """GQALaboratuvari motorunun benchmark raporunu test eder."""
    lab = GQALaboratuvari(dim=64, num_q_heads=8, katman_sayisi=4, cihaz=torch.device("cpu"))
    rapor = lab.gecikme_ve_throughput_olc(batch_size=2, seq_len=16, iterasyon=3)
    assert len(rapor) == 3
    for isim, metrikler in rapor.items():
        assert "p50_ms" in metrikler
        assert "throughput_tps" in metrikler
        assert metrikler["p50_ms"] > 0


def test_gqa_gorsellestirici_pano():
    """6 panelli GQA teşhis panosunun oluşturulduğunu test eder."""
    gorsellestirici = GQAGorsellestirici(dpi=100)
    ornek_gecikme = {
        "MHA (32 Q-Heads, 32 KV-Heads)": {"p50_ms": 4.5, "throughput_tps": 910000},
        "GQA (32 Q-Heads, 8 KV-Heads)": {"p50_ms": 2.6, "throughput_tps": 1570000},
        "MQA (32 Q-Heads, 1 KV-Head)": {"p50_ms": 2.1, "throughput_tps": 1950000},
    }
    ornek_bellek = {
        "MHA": [64.0, 128.0, 256.0, 512.0, 1024.0],
        "GQA": [16.0, 32.0, 64.0, 128.0, 256.0],
        "MQA": [2.0, 4.0, 8.0, 16.0, 32.0],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_gqa_paneli.png")
        gorsellestirici.pano_olustur(ornek_gecikme, ornek_bellek, kayit_yolu=kayit_yolu)
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
