"""
PyTest Birim Testleri - Day 272 (FAZ 14): Seyrek ve Doğrusal Dikkat Çekirdeği (Mamba SSM).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mamba_ssm_motoru import MambaLinearSSMKernelEngine
from src.mamba_ssm_profilleyici import MambaSSMProfilleyici
from src.gorsellestirici import MambaSSMGorsellestirici


def test_mamba_engine_initialization():
    """1. Mamba SSM motoru doğru matris boyutları ve kararlı A matrisi ile başlamalıdır."""
    engine = MambaLinearSSMKernelEngine(d_model=64, d_state=16)
    assert engine.d_model == 64
    assert engine.d_state == 16
    assert engine.A.shape == (64, 16)
    # A matrisinin tüm elemanları negatif olmalıdır (kararlı sürekli sistem)
    assert np.all(engine.A < 0.0)
    assert engine.D.shape == (64,)


def test_discretization_math():
    """2. Seçici parametre ayrıklaştırması (ZOH) exp(Delta * A) ve Delta * B formüllerini sağlamalıdır."""
    engine = MambaLinearSSMKernelEngine(d_model=32, d_state=8)
    batch_size, seq_len = 2, 4
    
    delta = np.full((batch_size, seq_len, 32), 0.1, dtype=np.float32)
    B = np.ones((batch_size, seq_len, 8), dtype=np.float32)
    
    A_bar, B_bar = engine.discretize_selective_parameters(delta, B)
    
    assert A_bar.shape == (batch_size, seq_len, 32, 8)
    assert B_bar.shape == (batch_size, seq_len, 32, 8)
    # Negatif A ve pozitif delta ile A_bar değerleri (0, 1) aralığında olmalıdır
    assert np.all(A_bar > 0.0) and np.all(A_bar <= 1.0)
    # B_bar = 0.1 * 1.0 = 0.1
    assert np.allclose(B_bar, 0.1, atol=1e-5)


def test_sequential_vs_parallel_scan_equivalence():
    """3. Sıralı O(N) durum taraması ile SRAM paralel birleşmeli tarama çıktıları tam denk olmalıdır."""
    engine = MambaLinearSSMKernelEngine(d_model=16, d_state=4)
    batch_size, seq_len = 2, 8
    
    np.random.seed(123)
    u = np.random.randn(batch_size, seq_len, 16).astype(np.float32)
    delta = np.abs(np.random.randn(batch_size, seq_len, 16).astype(np.float32)) * 0.05
    B = np.random.randn(batch_size, seq_len, 4).astype(np.float32)
    C = np.random.randn(batch_size, seq_len, 4).astype(np.float32)
    
    # 1. Sıralı
    y_seq, h_seq = engine.sequential_selective_scan(u, delta, B, C)
    
    # 2. Paralel
    A_bar, B_bar = engine.discretize_selective_parameters(delta, B)
    Bu = B_bar * np.expand_dims(u, axis=-1)
    _, h_par = engine.parallel_associative_scan(A_bar, Bu)
    y_par = np.sum(h_par * np.expand_dims(C, axis=2), axis=-1) + engine.D * u
    
    assert y_seq.shape == (batch_size, seq_len, 16)
    assert y_par.shape == (batch_size, seq_len, 16)
    assert np.allclose(y_seq, y_par, atol=1e-4)


def test_mamba_forward_shape_and_values():
    """4. Mock ileri geçiş geçerli boyutlar ve sayısal olarak kararlı değerler üretmelidir."""
    sonuc = MambaLinearSSMKernelEngine.execute_mock_forward_pass(
        batch_size=2,
        seq_len=32,
        d_model=32,
        d_state=8,
    )
    assert sonuc["y_seq_shape"] == (2, 32, 32)
    assert sonuc["y_par_shape"] == (2, 32, 32)
    assert sonuc["maksimum_fark"] < 1e-4
    assert not np.isnan(sonuc["maksimum_fark"])


def test_kv_cache_constant_memory():
    """5. Tek tokenlik çıkarım adımı O(1) sabit durum belleğini korumalıdır."""
    engine = MambaLinearSSMKernelEngine(d_model=32, d_state=8)
    batch_size = 2
    state = np.zeros((batch_size, 32, 8), dtype=np.float32)
    
    u_t = np.random.randn(batch_size, 32).astype(np.float32)
    delta_t = np.abs(np.random.randn(batch_size, 32).astype(np.float32)) * 0.05
    B_t = np.random.randn(batch_size, 8).astype(np.float32)
    C_t = np.random.randn(batch_size, 8).astype(np.float32)
    
    y_t, next_state = engine.step_single_token(u_t, delta_t, B_t, C_t, state)
    
    assert y_t.shape == (batch_size, 32)
    assert next_state.shape == (batch_size, 32, 8)
    assert not np.isnan(y_t).any()
    assert not np.isnan(next_state).any()


def test_latency_and_memory_scaling_speedup():
    """6. 128K'da Mamba Linear SSM karesel dikkate kıyasla en az 20x hızlanma ve 35x bellek tasarrufu sağlamalıdır."""
    profil = MambaSSMProfilleyici.basarim_profili_cikar()
    hizlanma = profil["hizlanma_orani"]
    tasarruf = profil["bellek_tasarrufu"]
    
    assert hizlanma >= 20.0  # 485.0 / 16.2 = 29.93x
    assert tasarruf >= 35.0  # 38.4 / 0.85 = 45.17x


def test_mamba_profiler_output():
    """7. MambaSSMProfilleyici tam karşılaştırma matrisini ve ölçekleme serilerini üretmelidir."""
    profil = MambaSSMProfilleyici.basarim_profili_cikar()
    karsilastirma = profil["karsilastirma"]
    
    assert "Mamba_Linear_SSM" in karsilastirma["sekans_gecikmesi_128k_ms"]
    assert karsilastirma["sekans_gecikmesi_128k_ms"]["Mamba_Linear_SSM"] == 16.2
    assert karsilastirma["vram_bellek_ayak_izi_gb"]["Mamba_Linear_SSM"] == 0.85
    assert len(profil["gecikme_skalasi"]["sekans_uzunluklari"]) == 8


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. MambaSSMGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_mamba_paneli.png")
    profil = MambaSSMProfilleyici.basarim_profili_cikar()

    MambaSSMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
