"""
Tesla FSD 8-Kamera Görüş Geometrisi ve İğne Deliği Modeli
=========================================================
Bu modül; Tesla HW3/HW4 mimarisindeki 8 adet kameranın içsel (Intrinsics K),
dışsal (Extrinsics [R|t]) matrislerini, Brown-Conrady distorsiyon düzeltmesini
ve 3D dünya koordinatlarından piksel düzlemlerine çoklu projeksiyonu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraModel:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    pos_ego_m: np.ndarray       # [X, Y, Z] metre cinsinden araç merkezi
    yaw_deg: float              # Derece (0 = ileri, +90 = sol, -90 = sağ, 180 = geri)
    pitch_deg: float
    roll_deg: float
    dist_k1: float = -0.05
    dist_k2: float = 0.01
    dist_p1: float = 0.001
    dist_p2: float = 0.001

    def get_intrinsics_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def get_rotation_matrix(self) -> np.ndarray:
        # Yaw (Z), Pitch (Y), Roll (X) Euler dönüşümü
        cy = np.cos(np.radians(self.yaw_deg))
        sy = np.sin(np.radians(self.yaw_deg))
        cp = np.cos(np.radians(self.pitch_deg))
        sp = np.sin(np.radians(self.pitch_deg))
        cr = np.cos(np.radians(self.roll_deg))
        sr = np.sin(np.radians(self.roll_deg))

        # Kamera koordinat sistemi: Z ileri (optik eksen), X sağ, Y aşağı
        # Ego araç koordinat sistemi: X ileri, Y sol, Z yukarı
        # Temel dönüşüm R_ego_to_cam
        R_z = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_y = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_x = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        
        # Standart optik eksen hizalama (Araç X -> Cam Z, Araç -Y -> Cam X, Araç -Z -> Cam Y)
        R_align = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ], dtype=np.float64)

        return R_align @ R_z @ R_y @ R_x

    def project_point_3d(self, pt_ego: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        3D Ego koordinatındaki bir noktayı kamera piksel düzlemine izdüşürür.
        Dönüş: ((u, v), Z_cam) veya nokta görüş açısı dışındaysa (None, Z_cam).
        """
        R = self.get_rotation_matrix()
        t = self.pos_ego_m
        
        # Kamera koordinatına ötele ve döndür: P_cam = R @ (P_ego - t)
        p_cam = R @ (pt_ego - t)
        z_cam = float(p_cam[2])

        if z_cam <= 0.2:  # Kameranın arkasında veya çok yakın
            return None, z_cam

        # Normalleştirilmiş koordinatlar
        x_norm = p_cam[0] / z_cam
        y_norm = p_cam[1] / z_cam

        # Brown-Conrady distorsiyonu uygula
        r2 = x_norm**2 + y_norm**2
        radial = 1.0 + self.dist_k1 * r2 + self.dist_k2 * (r2**2)
        x_dist = x_norm * radial + 2.0 * self.dist_p1 * x_norm * y_norm + self.dist_p2 * (r2 + 2.0 * x_norm**2)
        y_dist = y_norm * radial + self.dist_p1 * (r2 + 2.0 * y_norm**2) + 2.0 * self.dist_p2 * x_norm * y_norm

        # Piksel koordinatı (u, v)
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy

        if 0 <= u < self.width and 0 <= v < self.height:
            return (float(u), float(v)), z_cam
        return None, z_cam


class Tesla8CameraVisionRig:
    """
    Tesla FSD 8-Kamera Donanım Konfigürasyonu ve 360° Görüş Motoru.
    """
    def __init__(self):
        w, h = 1280, 960
        self.cameras: List[CameraModel] = [
            # 1. Front Main (50° FOV)
            CameraModel("Front_Main", 1200.0, 1200.0, w/2, h/2, w, h, np.array([2.0, 0.0, 1.35]), 0.0, -2.0, 0.0),
            # 2. Front Narrow (35° FOV)
            CameraModel("Front_Narrow", 1800.0, 1800.0, w/2, h/2, w, h, np.array([2.0, 0.0, 1.35]), 0.0, -1.0, 0.0),
            # 3. Front Wide (120° FOV)
            CameraModel("Front_Wide", 600.0, 600.0, w/2, h/2, w, h, np.array([2.0, 0.0, 1.35]), 0.0, -3.0, 0.0),
            # 4. Left Pillar (90° FOV, sol çapraz)
            CameraModel("Left_Pillar", 900.0, 900.0, w/2, h/2, w, h, np.array([1.2, 0.9, 1.2]), 60.0, -2.0, 0.0),
            # 5. Right Pillar (90° FOV, sağ çapraz)
            CameraModel("Right_Pillar", 900.0, 900.0, w/2, h/2, w, h, np.array([1.2, -0.9, 1.2]), -60.0, -2.0, 0.0),
            # 6. Left Repeater (Geri Sol Kör Nokta)
            CameraModel("Left_Repeater", 900.0, 900.0, w/2, h/2, w, h, np.array([0.5, 0.95, 0.8]), 140.0, -1.0, 0.0),
            # 7. Right Repeater (Geri Sağ Kör Nokta)
            CameraModel("Right_Repeater", 900.0, 900.0, w/2, h/2, w, h, np.array([0.5, -0.95, 0.8]), -140.0, -1.0, 0.0),
            # 8. Rear View (Geri Görüş)
            CameraModel("Rear_View", 800.0, 800.0, w/2, h/2, w, h, np.array([-2.2, 0.0, 0.9]), 180.0, -5.0, 0.0)
        ]

    def project_3d_scene(self, points_ego: List[np.ndarray]) -> Dict[str, List[Dict[str, Any]]]:
        """
        3D sahnedeki noktaları 8 kameranın her birine izdüşürür.
        """
        results = {}
        for cam in self.cameras:
            cam_detections = []
            for pt in points_ego:
                pixel_coord, depth = cam.project_point_3d(pt)
                if pixel_coord is not None:
                    cam_detections.append({
                        "pt_ego": pt.tolist(),
                        "pixel_uv": pixel_coord,
                        "depth_m": depth
                    })
            results[cam.name] = cam_detections
        return results
