"""
PyTest Birim Testleri - Day 262 (FAZ 14): Özel NVIDIA Tensor Core GEMM Çekirdeği.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tensor_core_gemm_motoru import (
    NaiveGEMM,
    SharedMemoryTiledGEMM,
    TensorCoreWMMASimulator,
)
from src.tensor_core_gemm_profilleyici import TensorCoreProfilleyici
from src.gorsellestirici import TensorCoreGorsellestirici


def test_naive_gemm_accuracy():
    """1. NaiveGEMM referans matris çarpımını doğru hesaplamalıdır."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    c = NaiveGEMM.execute(a, b)
    expected = np.dot(a, b)
    assert np.allclose(c, expected, atol=1e-5)


def test_shared_memory_tiled_gemm_shape():
    """2. SharedMemoryTiledGEMM doğru boyutlarda matris üretmelidir."""
    engine = SharedMemoryTiledGEMM(block_m=16, block_n=16, block_k=8)
    a = np.random.randn(32, 24).astype(np.float32)
    b = np.random.randn(24, 40).astype(np.float32)
    c, _ = engine.execute(a, b)
    assert c.shape == (32, 40)


def test_shared_memory_tiled_gemm_accuracy():
    """3. SharedMemoryTiledGEMM çıktısı numpy.dot ile tam eşleşmelidir."""
    engine = SharedMemoryTiledGEMM(block_m=16, block_n=16, block_k=8)
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)
    c, _ = engine.execute(a, b)
    expected = np.dot(a, b)
    assert np.allclose(c, expected, atol=1e-4)


def test_tensor_core_wmma_simulator_accuracy():
    """4. TensorCoreWMMASimulator doğruluğu FP16 toleransında sağlamalıdır."""
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)
    c, _ = TensorCoreWMMASimulator.execute_wmma(a, b)
    expected = np.dot(a, b)
    assert np.allclose(c, expected, atol=0.05)


def test_arithmetic_intensity_calculation():
    """5. TensorCoreWMMASimulator FLOP/Byte aritmetik yoğunluğu doğru hesaplamalıdır."""
    a = np.random.randn(16, 16).astype(np.float32)
    b = np.random.randn(16, 16).astype(np.float32)
    _, stats = TensorCoreWMMASimulator.execute_wmma(a, b)
    assert stats["toplam_flops"] == 2.0 * 16 * 16 * 16
    assert stats["aritmetik_yogunluk_flop_per_byte"] > 0.0


def test_sram_bank_conflict_padding():
    """6. SharedMemoryTiledGEMM padding parametresini doğru raporlamalıdır."""
    engine = SharedMemoryTiledGEMM(padding=4)
    _, stats = engine.execute(np.eye(16, dtype=np.float32), np.eye(16, dtype=np.float32))
    assert stats["sram_padding"] == 4
    assert stats["bank_conflict_status"] == "ENGEL_ASILDI"


def test_tensor_core_profiler_output():
    """7. TensorCoreProfilleyici kıyaslama raporunu eksiksiz üretmelidir."""
    profil = TensorCoreProfilleyici.basarim_profili_cikar()
    assert "Tensor_Core_WMMA" in profil["karsilastirma"]["islem_hizi_tflops"]
    assert profil["karsilastirma"]["islem_hizi_tflops"]["Tensor_Core_WMMA"] == 142.5


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. TensorCoreGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_tensor_core_gemm_paneli.png")
    profil = TensorCoreProfilleyici.basarim_profili_cikar()

    TensorCoreGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
