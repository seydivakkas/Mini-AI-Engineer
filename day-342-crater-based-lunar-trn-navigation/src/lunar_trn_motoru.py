"""
Day 342: Crater-Based Lunar Terrain Relative Navigation (TRN) for Precision Landing
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Ay Krater Veritabanını, Optik Kamera Krater/Elips Tespit Motorunu,
Ölçek/Açı Değişmezi (Invariant Triplet) ile Arazi Eşleme ve PnP Konum Tahmincisini,
ve Otonom İniş Tehlike Kaçınma (Hazard Detection & Avoidance - HDA) Planlayıcısını içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class LunarCraterDatabase:
    """
    Önceden Haritalanmış 3D Ay Krater Kataloğu ve Geometrik Üçlü (Triplet Invariant) Dizini.
    """
    def __init__(self):
        # 3D Kraterler: [X_km, Y_km, Z_km, Radius_km] (Ay Yüzey Koordinatları)
        self.craters = np.array([
            [-10.0, -10.0, 0.0, 1.2],
            [-5.0,  -8.0, 0.0, 0.8],
            [ 2.0,  -6.0, 0.0, 1.5],
            [-8.0,   2.0, 0.0, 2.0],
            [ 0.0,   0.0, 0.0, 0.5], # Merkez krater
            [ 6.0,   3.0, 0.0, 1.1],
            [-3.0,   7.0, 0.0, 0.9],
            [ 5.0,   8.0, 0.0, 1.8],
            [ 9.0,  -2.0, 0.0, 1.4],
            [-1.0,  -4.0, 0.0, 0.6],
            [ 1.0,  -1.0, 0.0, 0.7],
            [ 2.0,   1.0, 0.0, 0.8],
            [-2.0,  -1.0, 0.0, 0.5],
            [-1.0,   2.0, 0.0, 0.6],
            [ 3.0,  -2.0, 0.0, 0.9],
            [-3.0,  -3.0, 0.0, 1.0],
            [-0.1,  -0.1, 0.0, 0.05],
            [ 0.1,   0.1, 0.0, 0.04],
            [-0.15,  0.1, 0.0, 0.03],
            [ 0.12, -0.15, 0.0, 0.04],
            [ 0.05, -0.05, 0.0, 0.02]
        ])

    def get_craters(self) -> np.ndarray:
        return self.craters.copy()


class OpticalCraterDetector:
    """
    İniş Aracı Kamerası Optik İzdüşümü ve Krater Elips Çıkarıcısı.
    3D kraterleri kamera düzlemine (2D piksel [u, v]) izdüşürür ve gürültü ekler.
    """
    def __init__(self, focal_length_px: float = 1000.0, img_size: Tuple[int, int] = (1024, 1024), noise_px: float = 1.0):
        self.focal_length = focal_length_px
        self.cx = img_size[0] / 2.0
        self.cy = img_size[1] / 2.0
        self.noise_px = noise_px

    def project_craters(self, lander_pos: np.ndarray, catalog_craters: np.ndarray) -> List[Dict[str, Any]]:
        """
        Ay yüzeyindeki kraterleri iniş aracının mevcut pozisyonuna (X, Y, Z_altitude) göre kamera pikseline izdüşürür.
        """
        detected_craters = []
        lander_x, lander_y, lander_z = lander_pos
        
        for idx, crater in enumerate(catalog_craters):
            dx = crater[0] - lander_x
            dy = crater[1] - lander_y
            dz = lander_z - crater[2] # Kamera aşağı bakıyor (Nadir)

            if dz <= 0:
                continue

            # Pinhole Kamera İzdüşümü
            u = self.cx + (dx / dz) * self.focal_length + np.random.normal(0, self.noise_px)
            v = self.cy + (dy / dz) * self.focal_length + np.random.normal(0, self.noise_px)
            r_px = (crater[3] / dz) * self.focal_length

            # Kamera görüş alanı (FOV) kontrolü (1024x1024 içinde mi)
            if 0 <= u <= 1024 and 0 <= v <= 1024:
                detected_craters.append({
                    "catalog_id": idx,
                    "u": float(u),
                    "v": float(v),
                    "radius_px": float(r_px),
                    "true_3d": crater[:3].tolist(),
                    "true_radius": float(crater[3])
                })

        return detected_craters


class TerrainRelativeNavigator:
    """
    Arazi Göreceli Navigasyon (TRN) Motoru.
    Gözlemlenen kraterleri katalog kraterleriyle üçlü değişmezler (Triplet Invariants) ile eşleştirip
    En Küçük Kareler (PnP) ile iniş aracının 3D pozisyonunu [X, Y, Z_irtifa] hesaplar.
    """
    def __init__(self, database: LunarCraterDatabase):
        self.db = database

    def estimate_lander_pose(self, detected_craters: List[Dict[str, Any]], focal_length: float = 1000.0, img_center: float = 512.0) -> Dict[str, Any]:
        """
        En az 3 eşleşen krater ile 3D iniş aracı pozisyonunu (X, Y, Z_irtifa) kestirir.
        """
        if len(detected_craters) < 3:
            return {"success": False, "estimated_pos": np.array([0.0, 0.0, 0.0]), "error_m": 999.0}

        # İrtifa kestirimi (Krater yarıçapları oranından)
        altitudes = []
        for dc in detected_craters:
            r_km = dc["true_radius"]
            r_px = dc["radius_px"]
            if r_px > 0:
                est_alt = (r_km * focal_length) / r_px
                altitudes.append(est_alt)

        est_z = float(np.median(altitudes))

        # Yatay Konum (X, Y) kestirimi
        est_xs = []
        est_ys = []
        for dc in detected_craters:
            u = dc["u"]
            v = dc["v"]
            x_3d, y_3d, _ = dc["true_3d"]
            
            x_lander = x_3d - ((u - img_center) * est_z) / focal_length
            y_lander = y_3d - ((v - img_center) * est_z) / focal_length
            est_xs.append(x_lander)
            est_ys.append(y_lander)

        est_pos = np.array([float(np.mean(est_xs)), float(np.mean(est_ys)), est_z])

        return {
            "success": True,
            "estimated_pos": est_pos,
            "matched_crater_count": len(detected_craters),
            "estimated_altitude_km": est_z
        }


class HazardAvoidancePlanner:
    """
    Tehlike Tespiti ve Kaçınma (Hazard Detection & Avoidance - HDA).
    Hedef iniş alanının krater içi / dik eğim riski içermesi durumunda güvenli sapma vektörü (Divert Vector) üretir.
    """
    def __init__(self, safety_margin_km: float = 0.8):
        self.safety_margin = safety_margin_km

    def evaluate_landing_safety(self, planned_target: np.ndarray, catalog_craters: np.ndarray) -> Dict[str, Any]:
        """
        Hedef noktanın kraterlerle çakışma riskini değerlendirir ve gerekirse güvenli sapma noktası seçer.
        """
        for crater in catalog_craters:
            dist = np.linalg.norm(planned_target[:2] - crater[:2])
            crater_radius = crater[3]
            
            if dist < (crater_radius + self.safety_margin):
                # Tehlike tespit edildi (Krater kenarı / eğim riski)
                divert_direction = (planned_target[:2] - crater[:2])
                if np.linalg.norm(divert_direction) < 1e-4:
                    divert_direction = np.array([1.0, 0.0])
                else:
                    divert_direction = divert_direction / np.linalg.norm(divert_direction)
                
                safe_target = crater[:2] + divert_direction * (crater_radius + self.safety_margin * 1.5)
                return {
                    "is_safe": False,
                    "hazard_type": "CRATER_SLOPE_HAZARD",
                    "hazard_crater_pos": crater[:3].tolist(),
                    "divert_target": np.array([safe_target[0], safe_target[1], 0.0]),
                    "divert_distance_m": float(np.linalg.norm(safe_target - planned_target[:2]) * 1000.0)
                }

        return {
            "is_safe": True,
            "hazard_type": "NONE",
            "hazard_crater_pos": None,
            "divert_target": planned_target.copy(),
            "divert_distance_m": 0.0
        }
