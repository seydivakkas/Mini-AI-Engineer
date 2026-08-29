"""
PyTest Birim Testleri - Day 270 (FAZ 14): PyTorch C++ / CUDA Custom Extension.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cuda_extension_motoru import PyTorchCUDAExtensionEngine
from src.cuda_extension_profilleyici import PyTorchExtensionProfilleyici
from src.gorsellestirici import PyTorchExtensionGorsellestirici


def test_silu_activation_correctness():
    """1. SiLU aktivasyonu doğru değerler üretmelidir."""
    x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    res = PyTorchCUDAExtensionEngine.silu(x)
    assert res[1] == 0.0
    assert abs(res[2] - 4.9665) < 0.01


def test_cuda_source_code_contains_vectorized_float4():
    """2. CUDA C kaynak kodu float4, __global__ ve silu içermelidir."""
    code = PyTorchCUDAExtensionEngine.CUDA_SOURCE_CODE
    assert "float4" in code
    assert "__global__" in code
    assert "__device__" in code


def test_cpp_source_code_contains_torch_checks():
    """3. ATen C++ kaynak kodu TORCH_CHECK ve PYBIND11_MODULE içermelidir."""
    code = PyTorchCUDAExtensionEngine.CPP_SOURCE_CODE
    assert "TORCH_CHECK" in code
    assert "PYBIND11_MODULE" in code
    assert "torch::Tensor" in code


def test_setup_py_contains_cuda_extension():
    """4. setup.py dosyası CUDAExtension ve derleyici optimizasyon bayraklarını içermelidir."""
    code = PyTorchCUDAExtensionEngine.SETUP_PY_CODE
    assert "CUDAExtension" in code
    assert "-O3" in code
    assert "--use_fast_math" in code


def test_fused_swiglu_shape():
    """5. Fused SwiGLU doğru tensör boyutunu üretmelidir."""
    x1 = np.random.randn(64, 128).astype(np.float32)
    x2 = np.random.randn(64, 128).astype(np.float32)
    out, stats = PyTorchCUDAExtensionEngine.forward_fused_swiglu(x1, x2)
    assert out.shape == (64, 128)
    assert stats["cuda_kernel_sayisi"] == 1


def test_fused_swiglu_mathematical_identity():
    """6. Fused SwiGLU çıktısı referans formülle özdeş olmalıdır."""
    np.random.seed(42)
    x1 = np.random.randn(32, 64).astype(np.float32)
    x2 = np.random.randn(32, 64).astype(np.float32)
    out_fused, _ = PyTorchCUDAExtensionEngine.forward_fused_swiglu(x1, x2)
    ref = (x1 / (1.0 + np.exp(-x1))) * x2
    assert np.allclose(out_fused, ref, atol=1e-5)


def test_cuda_extension_profiler_output():
    """7. PyTorchExtensionProfilleyici 3'lü karşılaştırma raporunu üretmelidir."""
    profil = PyTorchExtensionProfilleyici.basarim_profili_cikar()
    assert "Custom_CUDA_Extension" in profil["karsilastirma"]["cekirdek_gecikmesi_us"]
    assert profil["karsilastirma"]["cekirdek_gecikmesi_us"]["Custom_CUDA_Extension"] == 2.1


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PyTorchExtensionGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_cuda_extension_paneli.png")
    profil = PyTorchExtensionProfilleyici.basarim_profili_cikar()

    PyTorchExtensionGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
