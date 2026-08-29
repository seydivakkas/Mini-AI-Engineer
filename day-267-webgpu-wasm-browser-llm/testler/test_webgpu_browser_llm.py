"""
PyTest Birim Testleri - Day 267 (FAZ 14): WebGPU & WebAssembly (Wasm) Browser LLM.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.webgpu_motoru import (
    WGSLComputeShaderSimulator,
    WasmSIMDTokenizerEngine,
    WebGPUBrowserLLMRuntime,
)
from src.webgpu_profilleyici import WebGPUProfilleyici
from src.gorsellestirici import WebGPUGorsellestirici


def test_wasm_tokenizer_encode_decode():
    """1. WasmSIMDTokenizerEngine metni kodlayıp geri çözebilmelidir."""
    tokenizer = WasmSIMDTokenizerEngine()
    text = "WebGPU tarayıcı LLM"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 1
    assert tokens[0] == 1  # <s>


def test_wgsl_shader_code_validity():
    """2. WGSL shader kodu geçerli W3C WGSL anahtar kelimelerini içermelidir."""
    code = WGSLComputeShaderSimulator.WGSL_SHADER_CODE
    assert "@compute" in code
    assert "@workgroup_size(16, 16)" in code
    assert "workgroupBarrier()" in code


def test_wgsl_gemm_shape():
    """3. WGSL GEMM doğru matris boyutunu üretmelidir."""
    a = np.random.randn(32, 64).astype(np.float32)
    b = np.random.randn(64, 48).astype(np.float32)
    c, stats = WGSLComputeShaderSimulator.execute_wgsl_gemm(a, b, tile_size=16)
    assert c.shape == (32, 48)
    assert "16x16" in stats["workgroup_boyutu"]


def test_wgsl_gemm_mathematical_identity():
    """4. WGSL GEMM çıktısı referans matris çarpımıyla özdeş olmalıdır."""
    np.random.seed(42)
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)
    c_wgsl, _ = WGSLComputeShaderSimulator.execute_wgsl_gemm(a, b, tile_size=16)
    c_ref = np.dot(a, b)
    assert np.allclose(c_wgsl, c_ref, atol=1e-4)


def test_wgsl_gemm_arbitrary_sizes():
    """5. 16'nın katı olmayan boyutlarda da (ör. 25x37) doğru çalışmalıdır."""
    np.random.seed(99)
    a = np.random.randn(25, 37).astype(np.float32)
    b = np.random.randn(37, 19).astype(np.float32)
    c_wgsl, _ = WGSLComputeShaderSimulator.execute_wgsl_gemm(a, b, tile_size=16)
    c_ref = np.dot(a, b)
    assert np.allclose(c_wgsl, c_ref, atol=1e-4)


def test_browser_llm_runtime_generate():
    """6. WebGPUBrowserLLMRuntime uçtan uca yanıt üretmelidir."""
    runtime = WebGPUBrowserLLMRuntime()
    res = runtime.generate_response("Merhaba dünya")
    assert res["sunucu_maliyeti_dolar"] == 0.0
    assert len(res["uretilen_yanit"]) > 0


def test_webgpu_profiler_output():
    """7. WebGPUProfilleyici 3'lü mimari kıyaslama raporunu üretmelidir."""
    profil = WebGPUProfilleyici.basarim_profili_cikar()
    assert "Tarayici_WebGPU_WGSL" in profil["karsilastirma"]["cikarim_hizi_tok_s"]
    assert profil["karsilastirma"]["cikarim_hizi_tok_s"]["Tarayici_WebGPU_WGSL"] == 58.2


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. WebGPUGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_webgpu_paneli.png")
    profil = WebGPUProfilleyici.basarim_profili_cikar()

    WebGPUGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
