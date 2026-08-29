"""
PyTest Birim Testleri - Day 251: İnsansı Robotik Bütünsel Hareket Kontrolü (Whole-Body Control & ZMP).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.humanoid_wbc_motoru import (
    LIPMDynamics,
    SupportPolygon,
    HierarchicalQPController,
)
from src.humanoid_wbc_profilleyici import HumanoidWBCProfilleyici
from src.gorsellestirici import HumanoidWBCGorsellestirici


def test_lipm_dynamics_init():
    """1. LIPMDynamics sarkaç frekansını doğru hesaplamalıdır."""
    lipm = LIPMDynamics(com_yukseklik_m=0.85, yercekimi=9.81)
    assert round(float(lipm.omega), 2) == 3.40


def test_lipm_compute_zmp():
    """2. compute_zmp p = x - (z_c/g)*a formülüyle ZMP üretmelidir."""
    lipm = LIPMDynamics(com_yukseklik_m=0.85, yercekimi=9.81)
    com_pos = np.array([0.10, 0.05])
    com_acc = np.array([0.0, 0.0])
    zmp = lipm.compute_zmp(com_pos, com_acc)
    assert np.allclose(zmp, com_pos)


def test_support_polygon_bounds():
    """3. SupportPolygon çift ayak sınırlarını simetrik kurmalıdır."""
    polygon = SupportPolygon(ayak_uzunlugu_m=0.22, ayak_genisligi_m=0.12, ayak_arasi_mesafe_m=0.20)
    assert polygon.x_min == -0.11
    assert polygon.x_max == 0.11
    assert polygon.y_min == -0.16
    assert polygon.y_max == 0.16


def test_support_polygon_is_stable():
    """4. is_zmp_stable iç ve dış noktaları doğru sınıflandırmalıdır."""
    polygon = SupportPolygon()
    assert polygon.is_zmp_stable(np.array([0.0, 0.0])) is True
    assert polygon.is_zmp_stable(np.array([0.5, 0.0])) is False


def test_support_polygon_margin():
    """5. compute_stability_margin poligon merkezinde pozitif marjin vermelidir."""
    polygon = SupportPolygon()
    marjin = polygon.compute_stability_margin(np.array([0.0, 0.0]))
    assert marjin > 0.10


def test_hierarchical_qp_controller_optimization():
    """6. optimize_wbc_step 80N dış kuvvete rağmen stabil ZMP üretmelidir."""
    lipm = LIPMDynamics(com_yukseklik_m=0.85)
    polygon = SupportPolygon()
    wbc = HierarchicalQPController(lipm=lipm, polygon=polygon)

    sonuc = wbc.optimize_wbc_step(
        com_pos=np.array([0.01, 0.01]),
        com_vel=np.array([0.05, 0.02]),
        hedef_com=np.array([0.0, 0.0]),
        dis_kuvvet_N=np.array([80.0, 20.0]),
    )
    assert sonuc["stabil_mi"] is True
    assert sonuc["guvenlik_marjini_cm"] > 0


def test_humanoid_wbc_profiler_output():
    """7. HumanoidWBCProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = HumanoidWBCProfilleyici.basarim_profili_cikar()
    assert "Hierarchical_QP_WBC" in profil["karsilastirma"]["denge_kararlilik_indeksi_yuzde"]
    assert profil["karsilastirma"]["denge_kararlilik_indeksi_yuzde"]["Hierarchical_QP_WBC"] == 99.2


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. HumanoidWBCGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_humanoid_wbc_paneli.png")
    profil = HumanoidWBCProfilleyici.basarim_profili_cikar()

    HumanoidWBCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
