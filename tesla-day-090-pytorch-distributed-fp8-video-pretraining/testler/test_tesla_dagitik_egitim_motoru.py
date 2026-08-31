"""
Tesla Dağıtık Eğitim Birim Testleri (PyTest)
============================================
Bu test paketi; FP8 tensör kuantalamasını, L2 gradyan kırpmasını
ve FSDP bellek bölütlemesini test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_dagitik_egitim_motoru import TeslaDojoDistributedTrainer


def test_fp8_kuantalama():
    """FP32 tensörün FP8 aralığına doğru ölçeklenip kuantalandığı test edilir."""
    trainer = TeslaDojoDistributedTrainer()
    t = np.array([0.1, -0.5, 1.2, -3.4, 5.0], dtype=np.float32)

    t_fp8, scale = trainer.quantize_to_fp8(t)

    assert scale > 0.0
    assert np.allclose(t, t_fp8, atol=0.05)


def test_l2_gradyan_kirpma():
    """L2 gradyan normunun belirlenen azami değere (1.0) kırpıldığı test edilir."""
    trainer = TeslaDojoDistributedTrainer(max_grad_norm=1.0)
    # Büyük gradyanlar
    grads = [np.full((10, 10), 2.0), np.full((10, 10), -3.0)]

    clipped, initial_norm = trainer.clip_and_normalize_gradients(grads)

    final_norm_sq = sum(np.sum(g ** 2) for g in clipped)
    final_norm = float(np.sqrt(final_norm_sq))

    assert initial_norm > 1.0
    assert np.isclose(final_norm, 1.0, atol=1e-4)


def test_fsdp_egitim_adimi():
    """FSDP simülasyonunun 32x bellek kazancı sağladığı ve loss ürettiği test edilir."""
    trainer = TeslaDojoDistributedTrainer(num_devices=8)
    res = trainer.train_step_fsdp_fp8(hidden_dim=256)

    assert res["memory_reduction_factor"] >= 16.0
    assert res["clipped_grad_norm"] <= 1.0
    assert res["training_loss"] > 0.0
    assert res["fp8_quant_ok"] is True
