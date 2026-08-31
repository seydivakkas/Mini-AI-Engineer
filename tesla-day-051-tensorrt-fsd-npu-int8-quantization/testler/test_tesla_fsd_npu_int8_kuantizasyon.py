"""
Tesla FSD NPU INT8 Kuantizasyon Birim Testleri (PyTest)
=======================================================
Bu test paketi; Simetrik INT8 kuantizasyon aralığını, SQNR sinyal kalitesini,
de-kuantizasyon doğruluğunu ve NPU Katman Birleştirmeyi (Layer Fusion) test eder.

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

from src.tesla_fsd_npu_int8_kuantizasyon import TeslaFSDNPUQuantizer


def test_simetrik_int8_aralik_ve_tip():
    """Kuantize edilen tensörün int8 tipinde ve [-128, 127] aralığında kaldığı test edilir."""
    quant = TeslaFSDNPUQuantizer()
    tensor_fp32 = np.array([-2.5, 0.0, 1.2, 3.8], dtype=np.float32)

    q_t, scale = quant.quantize_symmetric_int8(tensor_fp32)

    assert q_t.dtype == np.int8
    assert np.all(q_t >= -128) and np.all(q_t <= 127)
    assert scale > 0.0


def test_dekuantizasyon_ve_sqnr_kalitesi():
    """De-kuantize ağırlıkların SQNR değerinin 35 dB'nin üzerinde çıktığı test edilir."""
    quant = TeslaFSDNPUQuantizer()
    np.random.seed(42)
    weights = np.random.normal(0, 1.0, 1000).astype(np.float32)

    q_w, scale = quant.quantize_symmetric_int8(weights)
    deq_w = quant.dequantize_int8(q_w, scale)
    errs = quant.compute_sqnr_and_error(weights, deq_w)

    assert errs["sqnr_db"] > 35.0
    assert errs["mae"] < 0.02


def test_npu_katman_birlestirme_relu():
    """Fused Conv+BN+ReLU çıktısının negatif değer içermediği (ReLU doğruluk) test edilir."""
    quant = TeslaFSDNPUQuantizer()
    act = np.array([10, -5, 20], dtype=np.int8)
    w_int8 = np.array([2, -3, 1], dtype=np.int8)

    fused_out = quant.simulate_fused_conv_bn_relu(act, w_int8, scale=0.01)

    assert fused_out >= 0.0
