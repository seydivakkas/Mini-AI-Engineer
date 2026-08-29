"""
Multi-Head Latent Attention (MLA - DeepSeek V2/V3) Birim ve Entegrasyon Testleri (Day 103).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.latent_kv_cache import LatentKVCache
from src.mla_katmani import uygula_rope, MultiHeadLatentAttention
from src.karsilastirma_laboratuvari import MLALaboratuvari
from src.gorsellestirici import MLAGorsellestirici


def test_latent_kv_cache_guncelleme():
    """LatentKVCache tensör ekleme, indeksleme ve sıfırlama işlemlerini test eder."""
    cache = LatentKVCache(max_batch_size=4, max_seq_len=64, kv_latent_dim=32, rope_dim=16)
    c1 = torch.randn(2, 8, 32)
    r1 = torch.randn(2, 8, 16)

    aktif_c, aktif_r = cache.guncelle(c1, r1)
    assert aktif_c.shape == (2, 8, 32)
    assert aktif_r.shape == (2, 8, 16)
    assert cache.mevcut_uzunluk == 8

    # 1 token daha ekle
    c2 = torch.randn(2, 1, 32)
    r2 = torch.randn(2, 1, 16)
    aktif_c2, aktif_r2 = cache.guncelle(c2, r2)
    assert aktif_c2.shape == (2, 9, 32)
    assert aktif_r2.shape == (2, 9, 16)
    assert cache.mevcut_uzunluk == 9

    cache.sifirla()
    assert cache.mevcut_uzunluk == 0


def test_latent_kv_cache_bellek_hesabi():
    """LatentKVCache teorik bellek hesaplama fonksiyonunu doğrular."""
    # Katman=32, B=1, SeqLen=1024, dc=128, dR=32, FP16 (2 bayt)
    # Toplam: 32 * 1 * 1024 * (128 + 32) * 2 = 10,485,760 bayt = 10.0 MB
    cache = LatentKVCache(max_batch_size=1, max_seq_len=1024, kv_latent_dim=128, rope_dim=32, dtype=torch.float16)
    mb = cache.bellek_tuketimi_mb(katman_sayisi=32)
    assert round(mb, 1) == 10.0


def test_uygula_rope_donusumu():
    """RoPE rotasyon fonksiyonunun şekil ve tensör yapısını test eder."""
    x = torch.randn(2, 4, 16, 32)  # [B, H, S, D]
    x_rot = uygula_rope(x, seq_len_offset=0)
    assert x_rot.shape == (2, 4, 16, 32)

    # 3D girdi testi
    x_3d = torch.randn(2, 16, 32)  # [B, S, D]
    x_3d_rot = uygula_rope(x_3d, seq_len_offset=5)
    assert x_3d_rot.shape == (2, 16, 32)


def test_mla_ileri_gecis():
    """MultiHeadLatentAttention modülünün ileri geçişini test eder."""
    mla = MultiHeadLatentAttention(
        dim=64, num_heads=4, head_dim=16, kv_latent_dim=32, q_latent_dim=32, rope_dim=16
    )
    x = torch.randn(2, 8, 64)
    out, _ = mla(x)
    assert out.shape == (2, 8, 64)


def test_mla_adim_adim_cikarim_ve_cache():
    """MLA'nın LatentKVCache ile adım adım otoregresif çıkarımını test eder."""
    mla = MultiHeadLatentAttention(
        dim=64, num_heads=4, head_dim=16, kv_latent_dim=32, q_latent_dim=32, rope_dim=16
    )
    cache = LatentKVCache(max_batch_size=2, max_seq_len=32, kv_latent_dim=32, rope_dim=16)

    # 1. Prefill Aşaması (6 token)
    x_prompt = torch.randn(2, 6, 64)
    out1, cache = mla(x_prompt, latent_cache=cache, is_causal=True)
    assert out1.shape == (2, 6, 64)
    assert cache.mevcut_uzunluk == 6

    # 2. Decode Aşaması (1 token)
    x_token = torch.randn(2, 1, 64)
    out2, cache = mla(x_token, latent_cache=cache, is_causal=False)
    assert out2.shape == (2, 1, 64)
    assert cache.mevcut_uzunluk == 7


def test_mla_matris_boyutlari_ve_projeksiyonlar():
    """MLA projeksiyon matrislerinin düşük dereceli rank boyutlarını doğrular."""
    mla = MultiHeadLatentAttention(
        dim=128, num_heads=8, head_dim=32, kv_latent_dim=64, q_latent_dim=64, rope_dim=16
    )
    assert mla.w_dkv.weight.shape == (64, 128)
    assert mla.w_uk.weight.shape == (8 * 32, 64)
    assert mla.w_uv.weight.shape == (8 * 32, 64)
    assert mla.w_kr.weight.shape == (16, 128)


def test_mla_laboratuvari_karsilastirma():
    """MLALaboratuvari benchmark motorunun raporlarını doğrular."""
    lab = MLALaboratuvari(
        dim=64, num_heads=4, head_dim=16, kv_latent_dim=32, q_latent_dim=32, rope_dim=16, katman_sayisi=4
    )
    bellek = lab.kv_cache_bellek_karsilastirmasi(batch_size=2, dizi_uzunluklari=[128, 256])
    assert "DeepSeek MLA" in bellek
    assert "MHA (16 KV Kafa)" in bellek
    assert bellek["DeepSeek MLA"][0] < bellek["MHA (16 KV Kafa)"][0]

    gecikme = lab.gecikme_ve_throughput_olc(batch_size=2, seq_len=16, iterasyon=3)
    assert "DeepSeek MLA" in gecikme
    assert gecikme["DeepSeek MLA"]["p50_ms"] > 0


def test_mla_gorsellestirici_pano():
    """6 panelli MLA teşhis panosunun oluşturulduğunu test eder."""
    gorsellestirici = MLAGorsellestirici(dpi=100)
    ornek_gecikme = {"DeepSeek MLA": {"p50_ms": 3.2, "throughput_tps": 1280000}}
    ornek_bellek = {
        "MHA (16 KV Kafa)": [64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0],
        "GQA (4 KV Kafa)": [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
        "DeepSeek MLA": [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_mla_paneli.png")
        gorsellestirici.pano_olustur(ornek_gecikme, ornek_bellek, kayit_yolu=kayit_yolu)
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
