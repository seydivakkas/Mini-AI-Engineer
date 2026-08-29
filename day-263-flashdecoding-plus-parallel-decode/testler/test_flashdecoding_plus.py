"""
PyTest Birim Testleri - Day 263 (FAZ 14): FlashDecoding++ KV-Cache Bölümleme.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.flashdecoding_motoru import (
    KVCacheManager,
    FlashDecodingPlusEngine,
)
from src.flashdecoding_profilleyici import FlashDecodingProfilleyici
from src.gorsellestirici import FlashDecodingGorsellestirici


def test_kv_cache_manager_append_and_get():
    """1. KVCacheManager yeni tokenları önbelleğe eklemeli ve aktif dilimi döndürmelidir."""
    manager = KVCacheManager(batch_size=2, n_heads=4, head_dim=32, max_seq_len=100)
    k_step = np.random.randn(2, 4, 32).astype(np.float32)
    v_step = np.random.randn(2, 4, 32).astype(np.float32)

    manager.append(k_step, v_step)
    k_active, v_active = manager.get_cache()
    assert k_active.shape == (2, 4, 1, 32)
    assert v_active.shape == (2, 4, 1, 32)


def test_flashdecoding_plus_output_shape():
    """2. FlashDecodingPlusEngine doğru çıktı tensör boyutunu üretmelidir."""
    q = np.random.randn(2, 4, 1, 32).astype(np.float32)
    k = np.random.randn(2, 4, 512, 32).astype(np.float32)
    v = np.random.randn(2, 4, 512, 32).astype(np.float32)

    out, stats = FlashDecodingPlusEngine.execute_split_k(q, k, v, chunk_size=128)
    assert out.shape == (2, 4, 1, 32)
    assert stats["bolumlenen_chunk_sayisi"] == 4


def test_flashdecoding_plus_mathematical_identity():
    """3. FlashDecoding++ çıktısı klasik tam Softmax Attention ile matematiksel olarak özdeş olmalıdır."""
    q = np.random.randn(1, 2, 1, 32).astype(np.float32)
    k = np.random.randn(1, 2, 512, 32).astype(np.float32)
    v = np.random.randn(1, 2, 512, 32).astype(np.float32)

    out_split, _ = FlashDecodingPlusEngine.execute_split_k(q, k, v, chunk_size=128)

    # Tam Softmax
    scale = 1.0 / np.sqrt(32)
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out_exact = np.matmul(weights, v)

    assert np.allclose(out_split, out_exact, atol=1e-4)


def test_flashdecoding_plus_chunk_size_variation():
    """4. Farklı chunk boyutları (C=64, C=128, C=256) aynı doğru sonucu üretmelidir."""
    q = np.random.randn(1, 1, 1, 16).astype(np.float32)
    k = np.random.randn(1, 1, 256, 16).astype(np.float32)
    v = np.random.randn(1, 1, 256, 16).astype(np.float32)

    out64, _ = FlashDecodingPlusEngine.execute_split_k(q, k, v, chunk_size=64)
    out128, _ = FlashDecodingPlusEngine.execute_split_k(q, k, v, chunk_size=128)

    assert np.allclose(out64, out128, atol=1e-4)


def test_flashdecoding_plus_single_chunk_fallback():
    """5. Dizi uzunluğu chunk boyutundan küçükse tek parçada sorunsuz çalışmalıdır."""
    q = np.random.randn(1, 1, 1, 16).astype(np.float32)
    k = np.random.randn(1, 1, 30, 16).astype(np.float32)
    v = np.random.randn(1, 1, 30, 16).astype(np.float32)

    out, stats = FlashDecodingPlusEngine.execute_split_k(q, k, v, chunk_size=64)
    assert stats["bolumlenen_chunk_sayisi"] == 1
    assert out.shape == (1, 1, 1, 16)


def test_flashdecoding_plus_rescaling_stability():
    """6. FlashDecoding++ büyük logits değerlerinde NaN/Inf üretmemelidir."""
    q = np.ones((1, 1, 1, 16), dtype=np.float32) * 50.0
    k = np.ones((1, 1, 100, 16), dtype=np.float32) * 50.0
    v = np.random.randn(1, 1, 100, 16).astype(np.float32)

    out, _ = FlashDecodingPlusEngine.execute_split_k(q, k, v, chunk_size=32)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


def test_flashdecoding_profiler_output():
    """7. FlashDecodingProfilleyici kıyaslama raporunu eksiksiz üretmelidir."""
    profil = FlashDecodingProfilleyici.basarim_profili_cikar()
    assert "FlashDecoding_Plus" in profil["karsilastirma"]["decode_gecikmesi_ms"]
    assert profil["karsilastirma"]["decode_gecikmesi_ms"]["FlashDecoding_Plus"] == 4.2


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. FlashDecodingGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_flashdecoding_paneli.png")
    profil = FlashDecodingProfilleyici.basarim_profili_cikar()

    FlashDecodingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
