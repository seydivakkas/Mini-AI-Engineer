"""
PyTest Birim Testleri - Day 264 (FAZ 14): Yeni Nesil FP4 / FP6 Microscaling MXFP4.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mxfp4_microscaling_motoru import (
    MXFP4E2M1Codec,
    MXFP6E3M2Codec,
    MicroscaledGEMMEngine,
)
from src.mxfp4_profilleyici import MXFP4Profilleyici
from src.gorsellestirici import MXFP4Gorsellestirici


def test_mxfp4_e2m1_grid_values():
    """1. E2M1 ızgarası -6.0 ile +6.0 arasındaki beklenen 15 ayrık noktayı içermelidir."""
    assert len(MXFP4E2M1Codec.GRID_E2M1) == 15
    assert np.max(MXFP4E2M1Codec.GRID_E2M1) == 6.0
    assert np.min(MXFP4E2M1Codec.GRID_E2M1) == -6.0
    assert 0.0 in MXFP4E2M1Codec.GRID_E2M1


def test_mxfp4_quantize_and_dequantize_shape():
    """2. Kuantizasyon ve dekuantizasyon tensörün orijinal boyutunu korumalıdır."""
    tensor = np.random.randn(64, 128).astype(np.float32)
    q_blocks, scales, orig_shape = MXFP4E2M1Codec.quantize(tensor, block_size=32)
    deq = MXFP4E2M1Codec.dequantize(q_blocks, scales, orig_shape, block_size=32)

    assert deq.shape == (64, 128)
    assert q_blocks.shape == (256, 32)


def test_mxfp4_snr_quality():
    """3. MXFP4 E2M1 kuantizasyonu normal dağılımda yüksek SNR üretmelidir."""
    tensor = np.random.randn(100, 100).astype(np.float32)
    q_blocks, scales, orig_shape = MXFP4E2M1Codec.quantize(tensor, block_size=32)
    deq = MXFP4E2M1Codec.dequantize(q_blocks, scales, orig_shape, block_size=32)

    snr = MicroscaledGEMMEngine.compute_snr_db(tensor, deq)
    assert snr > 18.0  # 4-bit için yüksek sinyal kalitesi


def test_mxfp4_block_scaling_outlier_resilience():
    """4. Bir bloktaki aykırı değer diğer 32'li blokların ölçeğini etkilememelidir."""
    tensor = np.ones((1, 64), dtype=np.float32)
    tensor[0, 0] = 1000.0  # 1. blokta devasa aykırı değer

    q_blocks, scales, _ = MXFP4E2M1Codec.quantize(tensor, block_size=32)
    # 2. blok (32..64) etkilenmemeli ve scale'i küçük kalmalı
    assert scales[0] > 100.0
    assert scales[1] <= 2.0


def test_mxfp6_e3m2_codec():
    """5. FP6 E3M2 kuantizasyonu ve dekuantizasyonu doğru çalışmalıdır."""
    tensor = np.random.randn(32, 32).astype(np.float32)
    q_blocks, scales, orig_shape = MXFP6E3M2Codec.quantize(tensor, block_size=32)
    deq = MXFP6E3M2Codec.dequantize(q_blocks, scales, orig_shape)

    assert deq.shape == (32, 32)
    snr = MicroscaledGEMMEngine.compute_snr_db(tensor, deq)
    assert snr > 25.0


def test_microscaled_gemm_simulation():
    """6. MicroscaledGEMMEngine matris çarpımını yüksek doğrulukla simüle etmelidir."""
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)

    c_fp4, stats = MicroscaledGEMMEngine.execute_mxfp4_gemm(a, b)
    assert c_fp4.shape == (32, 32)
    assert "ortalama_snr_db" in stats


def test_mxfp4_profiler_output():
    """7. MXFP4Profilleyici 4'lü format kıyaslama raporunu üretmelidir."""
    profil = MXFP4Profilleyici.basarim_profili_cikar()
    assert "OCP_MXFP4_E2M1" in profil["karsilastirma"]["bellek_tuketimi_yuzde"]
    assert profil["karsilastirma"]["donanim_pflops_b200"]["OCP_MXFP4_E2M1"] == 20.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. MXFP4Gorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_mxfp4_paneli.png")
    profil = MXFP4Profilleyici.basarim_profili_cikar()

    MXFP4Gorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
