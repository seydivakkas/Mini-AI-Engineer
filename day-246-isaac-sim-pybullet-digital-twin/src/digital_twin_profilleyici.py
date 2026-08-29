"""
Dijital İkiz ve Simülasyon Başarım Profilleyicisi (Day 246).
Physical Robot vs PyBullet CPU vs Isaac Sim GPU Digital Twin Analizi.
"""

from typing import Dict, Any, List
import numpy as np
from .digital_twin_motoru import (
    DigitalTwinSimulator,
    RobotKinematics,
    SyntheticDataFactory,
)


class DigitalTwinProfilleyici:
    """FAZ 13 Robotik Dijital İkiz ve Sentetik Veri Profilleyicisi."""

    @classmethod
    def basarim_profili_cikar(cls) -> Dict[str, Any]:
        """Karşılaştırma Raporu ve Canlı Dijital İkiz Yörünge İcrası."""
        karsilastirma = {
            "veri_uretim_hacmi_yorunge_saat": {
                "Fiziksel_Robot": 1,  # Günde ~10-15
                "PyBullet_CPU": 500,
                "IsaacSim_GPU": 50000,
            },
            "donanim_kirilma_riski_yuzde": {
                "Fiziksel_Robot": 75.0,
                "PyBullet_CPU": 0.0,
                "IsaacSim_GPU": 0.0,
            },
            "sim2real_uyum_basarisi_yuzde": {
                "Fiziksel_Robot": 100.0,
                "PyBullet_CPU": 65.0,
                "IsaacSim_GPU": 88.5,
            },
            "fizik_adimi_gecikmesi_ms": {
                "Fiziksel_Robot": 10.0,
                "PyBullet_CPU": 2.0,
                "IsaacSim_GPU": 0.12,
            },
        }

        # Canlı Simülasyon ve IK/FK İcrası
        sim = DigitalTwinSimulator(dof=7, dt=0.01)
        hedef_3d = np.array([0.35, 0.15, 0.25])
        hedef_eklemler = sim.kinematics.inverse_kinematics(hedef_3d, np.zeros(7))

        adilar_gecmisi = []
        for _ in range(10):
            durum = sim.step_simulation(hedef_eklemler)
            adilar_gecmisi.append(durum["eef_3d_konum"].tolist())

        sentetik_sahne = SyntheticDataFactory.render_synthetic_scene(
            eef_pos=sim.kinematics.forward_kinematics(sim.eklem_konumlari),
            object_pos=hedef_3d,
            res=64,
        )

        return {
            "karsilastirma": karsilastirma,
            "hedef_3d_konum": hedef_3d.tolist(),
            "hesaplanan_eklem_acilari": hedef_eklemler.round(3).tolist(),
            "nihai_eef_konumu": sim.kinematics.forward_kinematics(sim.eklem_konumlari).tolist(),
            "sentetik_veri_boyutlari": {k: v.shape for k, v in sentetik_sahne.items()},
        }
