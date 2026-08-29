"""
PyTest Birim Testleri - Day 249: Dokunsal ve Kuvvet Sensörü Füzyon Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tactile_fusion_motoru import (
    GelSightTactileSensor,
    WristForceTorqueSensor,
    SlipDetectorAndGraspController,
    TactileGraspPipeline,
)
from src.tactile_profilleyici import TactileProfilleyici
from src.gorsellestirici import TactileGorsellestirici


def test_gelsight_contact_patch_shape():
    """1. GelSightTactileSensor (32, 32) boyutunda basınç haritası üretmelidir."""
    sensor = GelSightTactileSensor(res=32)
    patch = sensor.get_contact_patch(normal_kuvvet=3.0)
    assert patch["basinc_haritasi"].shape == (32, 32)
    assert patch["temas_alani_piksel"] > 0


def test_gelsight_pressure_increase_with_force():
    """2. Normal kuvvet arttığında temas alanı ve toplam baskı artmalıdır."""
    sensor = GelSightTactileSensor(res=32)
    patch_dusuk = sensor.get_contact_patch(normal_kuvvet=1.0)
    patch_yuksek = sensor.get_contact_patch(normal_kuvvet=8.0)
    assert patch_yuksek["temas_alani_piksel"] >= patch_dusuk["temas_alani_piksel"]
    assert patch_yuksek["toplam_baski_kuvveti"] > patch_dusuk["toplam_baski_kuvveti"]


def test_wrist_force_torque_wrench():
    """3. WristForceTorqueSensor 6 elemanlı wrench vektörü dönmelidir."""
    wrench = WristForceTorqueSensor.read_wrench(nesne_kutlesi_kg=0.5)
    assert len(wrench) == 6
    assert wrench[2] > 4.0  # Fz = m * g ~= 4.9N


def test_slip_detector_triggers():
    """4. detect_slip yüksek Ft/Fn oranında kayma bayrağını True dönmelidir."""
    ctrl = SlipDetectorAndGraspController(statik_surtunme_katsayisi=0.6)
    kayma_var, oran = ctrl.detect_slip(Fn=1.0, Ft=1.5)  # Oran: 1.5 > 0.6
    assert kayma_var is True
    assert oran == 1.5


def test_adaptive_force_controller_increases_on_slip():
    """5. Kayma algılandığında compute_adaptive_force normal kuvveti artırmalıdır."""
    ctrl = SlipDetectorAndGraspController()
    yeni_Fn = ctrl.compute_adaptive_force(mevcut_Fn=2.0, Ft=2.5)
    assert yeni_Fn > 2.0


def test_adaptive_force_controller_fragile_ceiling():
    """6. compute_adaptive_force kırılgan nesnelerde 12.0N tavanını aşmamalıdır."""
    ctrl = SlipDetectorAndGraspController(max_kirilma_kuvveti_N=12.0)
    yeni_Fn = ctrl.compute_adaptive_force(mevcut_Fn=11.5, Ft=10.0, is_fragile=True)
    assert yeni_Fn <= 12.0


def test_tactile_grasp_pipeline_simulation():
    """7. TactileGraspPipeline simülasyonu nesneyi kırmadan ve düşürmeden tamamlamalıdır."""
    pipeline = TactileGraspPipeline()
    res = pipeline.simulate_fragile_grasp(adim_sayisi=8)
    assert res["adim_sayisi"] == 8
    assert res["kirilma_oldu_mu"] is False
    assert res["dusurme_oldu_mu"] is False


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. TactileGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_tactile_paneli.png")
    profil = TactileProfilleyici.basarim_profili_cikar()

    TactileGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
