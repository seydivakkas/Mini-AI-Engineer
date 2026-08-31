"""
Tesla Kuşbakışı (Bird's Eye View - BEV), Homografi ve IPM Motoru
================================================================
Bu modül; Kamera perspektif görüntüsünü Düzlemsel Homografi ve Inverse
Perspective Mapping (IPM) kullanarak araç merkezli metrik Kuşbakışı (BEV)
koordinatlarına ($X_{\text{longitudinal}}, Y_{\text{lateral}}$) dönüştürür.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaBEVTransformer:
    """
    Inverse Perspective Mapping (IPM) ve BEV Dönüştürücüsü.
    """
    def __init__(
        self,
        camera_height_m: float = 1.35,
        pitch_deg: float = -2.0,
        roll_deg: float = 0.0,
        fx: float = 1200.0,
        fy: float = 1200.0,
        cx: float = 640.0,
        cy: float = 480.0,
        image_w: int = 1280,
        image_h: int = 960
    ):
        self.h = camera_height_m
        self.pitch_rad = np.radians(pitch_deg)
        self.roll_rad = np.radians(roll_deg)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.w = image_w
        self.image_h = image_h

        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Homografi Matrisi Hesaplama (Z_road = 0 düzlemi)
        # R_x(roll) @ R_y(pitch)
        cp = np.cos(self.pitch_rad)
        sp = np.sin(self.pitch_rad)
        cr = np.cos(self.roll_rad)
        sr = np.sin(self.roll_rad)

        # Kamera koordinat sistemi: Z ileri, X sağ, Y aşağı
        # Ego koordinat: X ileri, Y sol, Z yukarı (h yükseklikte)
        # Z_road = 0 varsayımıyla 3x3 Homografi
        R_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ], dtype=np.float64) @ np.array([
            [cp, 0, -sp],
            [0, 1, 0],
            [sp, 0, cp]
        ])

        r1 = R_cam[:, 0]
        r2 = R_cam[:, 1]
        t = np.array([0.0, 0.0, self.h])
        t_cam = R_cam @ (-t)

        self.H_road_to_cam = self.K @ np.column_stack((r1, r2, t_cam))
        self.H_cam_to_road = np.linalg.inv(self.H_road_to_cam)

    def pixel_to_bev(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Piksel koordinatını (u, v) metrik BEV yol koordinatına (X_ileri, Y_sol) çevirir.
        Ufuk çizgisinin üstündeki pikseller None döner.
        """
        # Ufuk çizgisi denetimi
        if v <= self.cy - (self.fy * np.tan(-self.pitch_rad)):
            return None

        p_img = np.array([u, v, 1.0], dtype=np.float64)
        p_road = self.H_cam_to_road @ p_img
        if abs(p_road[2]) < 1e-6:
            return None

        x_long = float(p_road[0] / p_road[2])
        y_lat = float(p_road[1] / p_road[2])

        if x_long <= 0.0 or x_long > 150.0:
            return None

        return (x_long, y_lat)

    def bev_to_pixel(self, x_long_m: float, y_lat_m: float) -> Optional[Tuple[float, float]]:
        """
        Metrik BEV koordinatını (X_ileri, Y_sol) kamera pikseline (u, v) çevirir.
        """
        if x_long_m <= 0.0:
            return None

        p_road = np.array([x_long_m, y_lat_m, 1.0], dtype=np.float64)
        p_img = self.H_road_to_cam @ p_road
        if abs(p_img[2]) < 1e-6:
            return None

        u = float(p_img[0] / p_img[2])
        v = float(p_img[1] / p_img[2])

        if 0 <= u < self.w and 0 <= v < self.image_h:
            return (u, v)
        return None

    def transform_lane_to_bev(self, lane_pixels: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        2D görüntüdeki şerit piksellerini BEV metrik koordinat listesine dönüştürür.
        """
        bev_pts = []
        for u, v in lane_pixels:
            pt = self.pixel_to_bev(u, v)
            if pt is not None:
                bev_pts.append(pt)
        return bev_pts
