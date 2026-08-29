"""
PyTest Birim Testleri - Day 246: Isaac Sim & PyBullet Dijital İkiz Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.digital_twin_motoru import (
    RobotKinematics,
    DigitalTwinSimulator,
    SyntheticDataFactory,
)
from src.digital_twin_profilleyici import DigitalTwinProfilleyici
from src.gorsellestirici import DigitalTwinGorsellestirici


def test_robot_kinematics_fk_shape():
    """1. forward_kinematics 3 elemanlı [x, y, z] dizisi dönmelidir."""
    kinematics = RobotKinematics()
    eef = kinematics.forward_kinematics(np.zeros(7))
    assert len(eef) == 3
    assert isinstance(eef, np.ndarray)


def test_robot_kinematics_ik_convergence():
    """2. inverse_kinematics hedef konuma yakınsamalıdır."""
    kinematics = RobotKinematics()
    hedef = np.array([0.3, 0.1, 0.2])
    eklemler = kinematics.inverse_kinematics(hedef, np.zeros(7), max_iter=30)
    fk = kinematics.forward_kinematics(eklemler)
    hata = np.linalg.norm(hedef - fk)
    assert len(eklemler) == 7
    assert hata < 0.5  # Kinematik yakınsama sınırı


def test_digital_twin_simulator_init():
    """3. DigitalTwinSimulator sıfır eklem konumuyla başlamalıdır."""
    sim = DigitalTwinSimulator(dof=7, dt=0.01)
    assert sim.dof == 7
    assert len(sim.eklem_konumlari) == 7
    assert sim.simulasyon_zamani == 0.0


def test_digital_twin_simulator_step():
    """4. step_simulation zamanı ve eklem durumunu güncellemelidir."""
    sim = DigitalTwinSimulator(dof=7, dt=0.01)
    durum = sim.step_simulation(np.ones(7) * 0.1)
    assert durum["zaman_sn"] == 0.01
    assert "eef_3d_konum" in durum
    assert len(durum["eklem_konumlari"]) == 7


def test_synthetic_data_factory_rgb_shape():
    """5. render_synthetic_scene RGB görüntüsünü (res, res, 3) olarak üretmelidir."""
    sentetik = SyntheticDataFactory.render_synthetic_scene(
        eef_pos=np.array([0.2, 0.2, 0.3]),
        object_pos=np.array([0.3, 0.1, 0.2]),
        res=32,
    )
    assert sentetik["rgb"].shape == (32, 32, 3)


def test_synthetic_data_factory_depth_and_mask():
    """6. Derinlik ve semantik maske beklenen sınıfları içermelidir."""
    sentetik = SyntheticDataFactory.render_synthetic_scene(
        eef_pos=np.array([0.1, 0.1, 0.3]),
        object_pos=np.array([0.2, 0.2, 0.2]),
        res=32,
    )
    assert sentetik["depth"].shape == (32, 32)
    assert sentetik["seg_mask"].shape == (32, 32)
    assert 1 in sentetik["seg_mask"]  # Robot maskesi
    assert 2 in sentetik["seg_mask"]  # Nesne maskesi


def test_digital_twin_profiler_output():
    """7. DigitalTwinProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = DigitalTwinProfilleyici.basarim_profili_cikar()
    assert "karsilastirma" in profil
    assert "veri_uretim_hacmi_yorunge_saat" in profil["karsilastirma"]
    assert profil["karsilastirma"]["veri_uretim_hacmi_yorunge_saat"]["IsaacSim_GPU"] == 50000


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DigitalTwinGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_digital_twin_paneli.png")
    profil = DigitalTwinProfilleyici.basarim_profili_cikar()

    DigitalTwinGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
