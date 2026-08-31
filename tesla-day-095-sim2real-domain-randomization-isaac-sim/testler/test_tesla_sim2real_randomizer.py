"""
Tesla Sim2Real Birim Testleri (PyTest)
======================================
Bu test paketi; dinamik alan rastgeleleştirmesini ve sıfır atışlı (Zero-Shot)
politika transfer başarısını test eder.

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

from src.tesla_sim2real_randomizer import TeslaSim2RealDomainRandomizer


def test_parametre_ornekleme_araliklari():
    """Rastgele örneklenen parametrelerin belirlenen sınırlar içinde kaldığı test edilir."""
    randomizer = TeslaSim2RealDomainRandomizer()
    params = randomizer.sample_randomized_parameters()

    # Sürtünme [0.40, 1.00]
    assert 0.40 <= params.ground_friction <= 1.00
    # Gecikme [0, 8] ms
    assert 0.0 <= params.latency_delay_ms <= 8.0
    # Kütle nominalin %85 ile %115'i arasında
    assert np.all(params.link_masses >= randomizer.nom_masses * 0.84)
    assert np.all(params.link_masses <= randomizer.nom_masses * 1.16)


def test_politika_dayanikligi_ve_sim2real_transfer():
    """100 rastgele simülasyon dünyasında başarı oranının %95 üzerinde olduğu test edilir."""
    randomizer = TeslaSim2RealDomainRandomizer()
    eval_res = randomizer.evaluate_policy_robustness(num_episodes=100)

    assert eval_res["num_episodes"] == 100
    assert eval_res["success_rate_pct"] >= 95.0
    assert eval_res["sim2real_ready"] is True
    assert eval_res["average_reward"] > 80.0
