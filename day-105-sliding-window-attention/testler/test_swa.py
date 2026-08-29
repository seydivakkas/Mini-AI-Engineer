"""
Mistral Sliding Window Attention (SWA) ve Rolling Buffer Cache Testleri (Day 105).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.rolling_buffer_cache import RollingBufferCache
from src.sliding_window_attention import olustur_bant_maskesi, repeat_kv, SlidingWindowAttention
from src.swa_laboratuvari import SWALaboratuvari
from src.gorsellestirici import SWAGorsellestirici


def test_rolling_buffer_cache_ekleme_ve_modulo():
    """RollingBufferCache modülünün dairesel halka indekslemesini test eder."""
    cache = RollingBufferCache(max_batch_size=2, window_size=4, num_kv_heads=2, head_dim=16)

    # 1. 3 token ekle (T < W)
    k1 = torch.randn(2, 2, 3, 16)
    v1 = torch.randn(2, 2, 3, 16)
    aktif_k, aktif_v = cache.guncelle(k1, v1)
    assert aktif_k.shape == (2, 2, 3, 16)
    assert cache.toplam_eklenen_token == 3

    # 2. 2 token daha ekle (toplam 5 > W=4) -> ilk slot'lar ezilmeli
    k2 = torch.randn(2, 2, 2, 16)
    v2 = torch.randn(2, 2, 2, 16)
    aktif_k2, aktif_v2 = cache.guncelle(k2, v2)
    assert aktif_k2.shape == (2, 2, 4, 16)  # Kapasite W=4 sabit
    assert cache.toplam_eklenen_token == 5

    cache.sifirla()
    assert cache.toplam_eklenen_token == 0


def test_rolling_buffer_cache_sabit_bellek():
    """RollingBufferCache'in dizi uzunluğundan bağımsız sabit bellek harcadığını doğrular."""
    # Katman=32, B=1, W=512, H_kv=8, head_dim=64, FP16
    # 2 * 32 * 1 * 8 * 512 * 64 * 2 = 33,554,432 bayt = 32.0 MB
    cache = RollingBufferCache(max_batch_size=1, window_size=512, num_kv_heads=8, head_dim=64, dtype=torch.float16)
    mb = cache.bellek_tuketimi_mb(katman_sayisi=32)
    assert round(mb, 1) == 32.0


def test_olustur_bant_maskesi_sekil_ve_degerler():
    """olustur_bant_maskesi fonksiyonunun nedensel bant sınırlarını doğrular."""
    maske = olustur_bant_maskesi(seq_len=6, window_size=3, device=torch.device("cpu"))
    assert maske.shape == (6, 6)

    # (i=0, j=0) geçerli (0.0)
    assert maske[0, 0] == 0.0
    # (i=0, j=1) gelecekte (-inf)
    assert maske[0, 1] == float("-inf")
    # (i=4, j=4) geçerli (0.0)
    assert maske[4, 4] == 0.0
    # (i=4, j=0) pencere dışı (4 - 0 = 4 >= 3) -> (-inf)
    assert maske[4, 0] == float("-inf")
    # (i=4, j=2) pencere içi (4 - 2 = 2 < 3) -> (0.0)
    assert maske[4, 2] == 0.0


def test_swa_prefill_ileri_gecis():
    """SlidingWindowAttention modülünün bant maskeli prefill ileri geçişini test eder."""
    swa = SlidingWindowAttention(dim=64, num_q_heads=4, num_kv_heads=2, window_size=4)
    x = torch.randn(2, 8, 64)
    out, _ = swa(x)
    assert out.shape == (2, 8, 64)


def test_swa_adim_adim_decode_ve_rolling_cache():
    """SWA katmanının RollingBufferCache ile W penceresinden daha uzun üretim yapmasını test eder."""
    swa = SlidingWindowAttention(dim=64, num_q_heads=4, num_kv_heads=2, window_size=4)
    cache = RollingBufferCache(max_batch_size=2, window_size=4, num_kv_heads=2, head_dim=16)

    # 10 adım token üret
    for step in range(10):
        x_token = torch.randn(2, 1, 64)
        out, cache = swa(x_token, rolling_cache=cache)
        assert out.shape == (2, 1, 64)

    assert cache.toplam_eklenen_token == 10


def test_etkin_alici_alan_hesaplayici():
    """SWA katman istifleme alıcı alan formülünü (L * W) test eder."""
    lab = SWALaboratuvari(dim=64, window_size=512, katman_sayisi=32, cihaz=torch.device("cpu"))
    alici_alan = lab.etkin_alici_alan_hesabi()
    assert alici_alan["maksimum_alici_alan"] == 32 * 512  # 16,384 token
    assert len(alici_alan["katman_bazli_alanlar"]) == 32


def test_swa_laboratuvari_karsilastirma():
    """SWALaboratuvari bellek ve throughput benchmark motorunu test eder."""
    lab = SWALaboratuvari(dim=64, window_size=64, katman_sayisi=4, cihaz=torch.device("cpu"))
    bellek = lab.kv_cache_bellek_karsilastirmasi(batch_size=2, dizi_uzunluklari=[64, 128, 256])
    assert len(bellek) == 2
    # 256 token'da SWA belleği Full Attention'dan çok daha küçük olmalı
    assert list(bellek.values())[1][-1] < list(bellek.values())[0][-1]

    gecikme = lab.gecikme_ve_throughput_olc(batch_size=2, seq_len=16, iterasyon=3)
    assert "Mistral SWA" in gecikme


def test_swa_gorsellestirici_pano():
    """6 panelli SWA teşhis panosunun oluşturulduğunu test eder."""
    gorsellestirici = SWAGorsellestirici(dpi=100)
    ornek_gecikme = {"Mistral SWA": {"p50_ms": 2.5, "throughput_tps": 1600000}}
    ornek_bellek = {
        "Full Causal Attention (O(S))": [64.0, 128.0, 256.0, 512.0, 1024.0],
        "Mistral SWA (W=512) (O(W))": [64.0, 64.0, 64.0, 64.0, 64.0],
    }
    ornek_alici = {
        "toplam_katman": 32,
        "pencere_boyutu": 512,
        "katman_bazli_alanlar": [l * 512 for l in range(1, 33)],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_swa_paneli.png")
        gorsellestirici.pano_olustur(
            ornek_gecikme, ornek_bellek, ornek_alici, dizi_uzunluklari=[512, 1024, 2048, 4096, 8192], kayit_yolu=kayit_yolu
        )
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
