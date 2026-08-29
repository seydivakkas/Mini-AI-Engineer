"""
Isaac Sim & PyBullet Robotik Dijital İkiz ve Sentetik Veri Motoru (Day 246).
İleri/Ters Kinematik (FK/IK), Katı Cisim Fiziği ve Çok Modlu RGB-D Sentetik Veri Üreticisi.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch


class RobotKinematics:
    """7-DoF Robot Kolu İleri ve Ters Kinematik (FK / IK) Çözücüsü."""

    def __init__(self, link_uzunluklari: List[float] = None):
        self.linkler = link_uzunluklari or [0.15, 0.20, 0.20, 0.15, 0.15, 0.10, 0.05]
        self.dof = len(self.linkler)

    def forward_kinematics(self, eklem_acilari: np.ndarray) -> np.ndarray:
        """Eklem açılarından uç nokta (EEF) 3D konumunu [x, y, z] hesaplar."""
        x = 0.0
        y = 0.0
        z = 0.1  # Taban yüksekliği
        kumulatif_yaw = 0.0
        kumulatif_pitch = 0.0

        for i in range(min(len(eklem_acilari), self.dof)):
            aci = eklem_acilari[i]
            if i % 2 == 0:
                kumulatif_yaw += aci
            else:
                kumulatif_pitch += aci

            L = self.linkler[i]
            x += L * np.cos(kumulatif_yaw) * np.cos(kumulatif_pitch)
            y += L * np.sin(kumulatif_yaw) * np.cos(kumulatif_pitch)
            z += L * np.sin(kumulatif_pitch)

        return np.array([round(float(x), 4), round(float(y), 4), round(float(z), 4)])

    def inverse_kinematics(self, hedef_pos: np.ndarray, baslangic_acilari: np.ndarray, lr: float = 1.0, max_iter: int = 50) -> np.ndarray:
        """Sönümlü En Küçük Kareler (DLS) ile hedef konuma eklem açılarını hesaplar."""
        acilari = baslangic_acilari.copy().astype(np.float64)
        if np.all(acilari == 0) or np.linalg.norm(acilari) < 0.1:
            acilari = np.array([0.8, -0.6, 0.4, 0.7, 0.0, 0.3, 0.0])  # Tekillikten kaçınma konfigürasyonu
        delta = 1e-4
        lambda_damp = 0.05

        for _ in range(max_iter):
            mevcut_pos = self.forward_kinematics(acilari)
            hata = hedef_pos - mevcut_pos
            if np.linalg.norm(hata) < 0.01:  # 1cm tolerans
                break

            # Sayısal Jakoben Hesabı (3 x dof)
            J = np.zeros((3, self.dof))
            for j in range(self.dof):
                acilari_arti = acilari.copy()
                acilari_arti[j] += delta
                pos_arti = self.forward_kinematics(acilari_arti)
                J[:, j] = (pos_arti - mevcut_pos) / delta

            # Damped Least Squares (DLS)
            JJT = np.dot(J, J.T) + (lambda_damp ** 2) * np.eye(3)
            J_dls = np.dot(J.T, np.linalg.inv(JJT))

            adim = lr * np.dot(J_dls, hata)
            acilari += adim
            acilari = np.clip(acilari, -np.pi, np.pi)

        return acilari


class DigitalTwinSimulator:
    """Robotik Dijital İkiz ve Katı Cisim Fizik Simülatörü."""

    def __init__(self, dof: int = 7, dt: float = 0.01):
        self.dof = dof
        self.dt = dt
        self.kinematics = RobotKinematics()
        self.eklem_konumlari = np.zeros(dof, dtype=np.float64)
        self.eklem_hizlari = np.zeros(dof, dtype=np.float64)
        self.simulasyon_zamani = 0.0

    def step_simulation(self, hedef_eklemler: np.ndarray) -> Dict[str, Any]:
        """PD Kontrolcü ve fizik integratörü ile bir zaman adımı (dt=0.01s) ilerler."""
        kp = 150.0  # Oransal kazanç
        kd = 20.0   # Türevsel kazanç

        hata = hedef_eklemler - self.eklem_konumlari
        tork = kp * hata - kd * self.eklem_hizlari
        tork = np.clip(tork, -50.0, 50.0)  # Tork sınırı

        # Basit Euler Entegrasyonu (Ivme = Tork / Kütle Ataleti)
        atalet = 0.5
        ivme = tork / atalet
        self.eklem_hizlari += ivme * self.dt
        self.eklem_konumlari += self.eklem_hizlari * self.dt
        self.simulasyon_zamani += self.dt

        eef_pos = self.kinematics.forward_kinematics(self.eklem_konumlari)

        return {
            "zaman_sn": round(self.simulasyon_zamani, 3),
            "eklem_konumlari": self.eklem_konumlari.copy(),
            "eklem_hizlari": self.eklem_hizlari.copy(),
            "eef_3d_konum": eef_pos,
        }


class SyntheticDataFactory:
    """Çok Modlu Sentetik RGB-D ve Segmentasyon Görüntü Fabrikası."""

    @classmethod
    def render_synthetic_scene(cls, eef_pos: np.ndarray, object_pos: np.ndarray, res: int = 64) -> Dict[str, np.ndarray]:
        """Simülasyondan fotogerçekçi sentetik RGB, Derinlik ve Semantik maske üretir."""
        # Sentetik RGB (Derinlik gradyanlı)
        rgb = np.ones((res, res, 3), dtype=np.float32) * 0.15  # Koyu gri masa arka planı
        # Robot EEF projeksiyonu
        cx, cy = int(res * (0.5 + eef_pos[0] * 0.5)), int(res * (0.5 + eef_pos[1] * 0.5))
        cx = np.clip(cx, 5, res - 6)
        cy = np.clip(cy, 5, res - 6)
        rgb[cy-4:cy+5, cx-4:cx+5] = [0.2, 0.7, 1.0]  # Mavi EEF

        # Hedef Nesne
        ox, oy = int(res * (0.5 + object_pos[0] * 0.5)), int(res * (0.5 + object_pos[1] * 0.5))
        ox = np.clip(ox, 5, res - 6)
        oy = np.clip(oy, 5, res - 6)
        rgb[oy-4:oy+5, ox-4:ox+5] = [1.0, 0.2, 0.2]  # Kırmızı Nesne

        # Sentetik Derinlik (Metre cinsinden mesafe)
        depth = np.ones((res, res), dtype=np.float32) * 1.5
        depth[cy-4:cy+5, cx-4:cx+5] = 0.8
        depth[oy-4:oy+5, ox-4:ox+5] = 0.75

        # Semantik Maske (0: Zemin, 1: Robot, 2: Nesne)
        seg_mask = np.zeros((res, res), dtype=np.int32)
        seg_mask[cy-4:cy+5, cx-4:cx+5] = 1
        seg_mask[oy-4:oy+5, ox-4:ox+5] = 2

        return {
            "rgb": rgb,
            "depth": depth,
            "seg_mask": seg_mask,
        }
