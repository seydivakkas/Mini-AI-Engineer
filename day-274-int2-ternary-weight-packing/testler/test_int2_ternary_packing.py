"""
PyTest Birim Testleri - Day 274 (FAZ 14): Bit Düzeyinde Paketleme (Bit-Packing: INT2 & Ternary).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bit_packing_motoru import BitPackingKernelEngine
from src.bit_packing_profilleyici import BitPackingProfilleyici
from src.gorsellestirici import BitPackingGorsellestirici


def test_int2_packing_shape_and_type():
    """1. 2-Bit paketleme her 16 elemanı 1 adet uint32 içinde sıkıştırmalıdır."""
    w = np.random.randint(0, 4, size=(64, 64), dtype=np.uint8)
    packed, shape = BitPackingKernelEngine.pack_int2_weights(w)
    assert packed.dtype == np.uint32
    assert len(packed) == (64 * 64) // 16
    assert shape == (64, 64)


def test_int2_unpack_roundtrip():
    """2. INT2 paketleme ve geri çözme orijinal tensörle bit düzeyinde birebir eşleşmelidir."""
    w = np.random.randint(0, 4, size=(32, 48), dtype=np.uint8)
    packed, shape = BitPackingKernelEngine.pack_int2_weights(w)
    unpacked = BitPackingKernelEngine.unpack_int2_weights(packed, shape)
    assert np.array_equal(w, unpacked)


def test_ternary_packing_roundtrip():
    """3. {-1, 0, 1} Ternary ağırlıklar UINT32 paketlemeden kayıpsız geri dönmelidir."""
    w = np.random.choice([-1, 0, 1], size=(128, 64)).astype(np.int8)
    packed, shape = BitPackingKernelEngine.pack_ternary_weights(w)
    unpacked = BitPackingKernelEngine.unpack_ternary_weights(packed, shape)
    assert np.array_equal(w, unpacked)


def test_fused_packed_gemm_math():
    """4. Fused unpack GEMM çekirdeği standart matris çarpımıyla tam matematiksel uyumda olmalıdır."""
    w = np.random.choice([-1, 0, 1], size=(64, 32)).astype(np.int8)
    packed, shape = BitPackingKernelEngine.pack_ternary_weights(w)
    x = np.random.randn(8, 64).astype(np.float32)
    scale = 0.125
    
    out_fused = BitPackingKernelEngine.fused_packed_gemm(x, packed, shape, scale=scale)
    out_expected = np.dot(x, w.astype(np.float32) * scale)
    
    assert out_fused.shape == (8, 32)
    assert np.allclose(out_fused, out_expected, atol=1e-5)


def test_compression_ratio_exact():
    """5. INT2 paketlenmiş tensör FP16'ya göre tam 8.0x bellek sıkıştırması sağlamalıdır."""
    res = BitPackingKernelEngine.execute_mock_packing_pipeline(matrix_rows=1024, matrix_cols=1024)
    assert res["tam_eslesme"] is True
    assert res["tasarruf_orani_fp16"] == 8.0


def test_profiler_output_and_speedup():
    """6. BitPackingProfilleyici 70B modelinde 8.0x VRAM tasarrufu ve 4.7x hızlanma raporlamalıdır."""
    profil = BitPackingProfilleyici.basarim_profili_cikar()
    assert profil["vram_tasarrufu"] == 8.0
    assert profil["hizlanma_orani"] > 4.5
    assert profil["karsilastirma"]["vram_ayak_izi_70b_gb"]["INT2_Ternary_Packed"] == 17.5


def test_bitfield_padding_handling():
    """7. 16'nın tam katı olmayan boyutlardaki diziler doğru padding ile hatasız çözülmelidir."""
    w = np.random.choice([-1, 0, 1], size=(17, 19)).astype(np.int8)  # 323 eleman (16'nın katı değil)
    packed, shape = BitPackingKernelEngine.pack_ternary_weights(w)
    unpacked = BitPackingKernelEngine.unpack_ternary_weights(packed, shape)
    assert unpacked.shape == (17, 19)
    assert np.array_equal(w, unpacked)


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. BitPackingGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_packing_paneli.png")
    profil = BitPackingProfilleyici.basarim_profili_cikar()

    BitPackingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
