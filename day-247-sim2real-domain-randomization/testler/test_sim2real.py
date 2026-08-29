"""
PyTest Birim Testleri - Day 247: Sim2Real Domain Randomization Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain_randomization_motoru import (
    VisualRandomizer,
    DynamicsRandomizer,
    ActionDelayInjector,
    Sim2RealEvaluator,
)
from src.sim2real_profilleyici import Sim2RealProfilleyici
from src.gorsellestirici import Sim2RealGorsellestirici


def test_visual_randomizer_shape_and_range():
    """1. randomize_image boyutları korumalı ve [0.0, 1.0] aralığında kalmalıdır."""
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    dr_img = VisualRandomizer.randomize_image(img, tohum=1)
    assert dr_img.shape == (32, 32, 3)
    assert np.min(dr_img) >= 0.0
    assert np.max(dr_img) <= 1.0


def test_visual_randomizer_variance():
    """2. Görsel rastgeleleştirme görüntüye varyans katmalıdır."""
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    dr_img = VisualRandomizer.randomize_image(img, tohum=2)
    fark = np.abs(dr_img - img)
    assert np.sum(fark) > 0.0


def test_dynamics_randomizer_parameter_keys():
    """3. sample_dynamics_parameters gerekli tüm fizik parametrelerini içermelidir."""
    param = DynamicsRandomizer.sample_dynamics_parameters(tohum=3)
    gerekli = ["surtunme_katsayisi", "kutle_carpani", "eklem_sonumleme", "tork_carpani"]
    for k in gerekli:
        assert k in param


def test_dynamics_randomizer_parameter_ranges():
    """4. Dinamik parametreler tanımlı fiziksel aralıklarda olmalıdır."""
    param = DynamicsRandomizer.sample_dynamics_parameters(tohum=4)
    assert 0.15 <= param["surtunme_katsayisi"] <= 1.25
    assert 0.80 <= param["kutle_carpani"] <= 1.20


def test_action_delay_injector():
    """5. ActionDelayInjector eylemleri gecikme adımı kadar ötelemelidir."""
    delay = ActionDelayInjector(kuyruk_boyutu=3)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([2.0, 0.0])
    delay.apply_delay(a1)
    delay.apply_delay(a2)
    out = delay.apply_delay(np.array([3.0, 0.0]), gecikme_adimi=1)
    assert out[0] == 2.0


def test_sim2real_evaluator_modes():
    """6. Full multimodal DR başarı oranı Naive Sim'den yüksek olmalıdır."""
    res_naive = Sim2RealEvaluator.evaluate_regime("naive_sim")
    res_full = Sim2RealEvaluator.evaluate_regime("full_multimodal_dr")
    assert res_full["basari_orani_yuzde"] > res_naive["basari_orani_yuzde"]
    assert res_full["ortalama_hata_cm"] < res_naive["ortalama_hata_cm"]


def test_sim2real_profiler_output():
    """7. Sim2RealProfilleyici 4 rejimi eksiksiz karşılaştırmalıdır."""
    profil = Sim2RealProfilleyici.basarim_profili_cikar()
    assert profil["rejim_sayisi"] == 4
    assert "Full_Multimodal_DR" in profil["karsilastirma"]["gercek_dunya_basari_yuzdesi"]
    assert profil["karsilastirma"]["gercek_dunya_basari_yuzdesi"]["Full_Multimodal_DR"] == 94.2


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. Sim2RealGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_sim2real_paneli.png")
    profil = Sim2RealProfilleyici.basarim_profili_cikar()

    Sim2RealGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
