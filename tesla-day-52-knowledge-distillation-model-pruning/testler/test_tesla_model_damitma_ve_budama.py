"""
Tesla Model Damıtma ve Budama Birim Testleri (PyTest)
=====================================================
Bu test paketi; Sıcaklık yumuşatmalı softmax'ı, Bilgi Damıtma (KD) kaybını
ve L1-Norm kanal budama seyreklik oranını test eder.

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

from src.tesla_model_damitma_ve_budama import TeslaKnowledgeDistiller


def test_sicaklik_yumusatmali_softmax():
    """T=4.0 sıcaklığında olasılıkların T=1.0'e göre daha yumuşak (daha yüksek entropili) olduğu test edilir."""
    distiller = TeslaKnowledgeDistiller()
    logits = np.array([5.0, 1.0, 0.0])

    p_hard = distiller.compute_soft_probabilities(logits, temperature=1.0)
    p_soft = distiller.compute_soft_probabilities(logits, temperature=4.0)

    assert p_hard[0] > p_soft[0]  # En büyük sınıfın baskınlığı yumuşatılır
    assert p_soft[1] > p_hard[1]  # İkincil sınıfların karanlık bilgisi (dark knowledge) açığa çıkar
    assert np.isclose(np.sum(p_soft), 1.0)


def test_bilgi_damitma_kaybi_hesabi():
    """KD kaybının hem yumuşak KL hem de sert CE kaybını pozitif olarak içerdiği test edilir."""
    distiller = TeslaKnowledgeDistiller(temperature=4.0, alpha=0.7)
    teacher_l = np.array([6.0, 2.0, 0.5])
    student_l = np.array([4.0, 1.5, 0.2])
    true_y = np.array([1.0, 0.0, 0.0])

    losses = distiller.compute_distillation_loss(teacher_l, student_l, true_y)

    assert losses["total_loss"] > 0.0
    assert losses["loss_soft_kd"] > 0.0
    assert losses["loss_hard_ce"] > 0.0


def test_l1_norm_kanal_budama_seyreklik():
    """%30 budama oranında 10 kanallı evrişim tensöründen tam 3 kanalın sıfırlandığı test edilir."""
    distiller = TeslaKnowledgeDistiller()
    weights = np.ones((10, 4, 3, 3), dtype=np.float32)
    # İlk 3 kanalı bilerek çok küçük yapalım
    weights[0:3] = 0.01

    pruned_w, mask, sparsity = distiller.prune_channels_l1_norm(weights, prune_ratio=0.3)

    assert int(np.sum(~mask)) == 3
    assert np.all(pruned_w[0:3] == 0.0)
    assert np.isclose(sparsity, 0.3)
