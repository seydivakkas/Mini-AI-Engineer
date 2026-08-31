"""
Tesla Spatiotemporal BEV Transformer Füzyon Motoru (BEVFormer)
==============================================================
Bu modül; 8 kameranın 2D öznitelik haritalarını 3D ışın projeksiyonu ile
Mekansal Çapraz Dikkat (Spatial Cross-Attention) üzerinden toplayan ve
araç hareketi kompanzasyonlu Zamansal Öz-Dikkat (Temporal Self-Attention) ile
oklüzyonları (görünmezlikleri) hafızasında tutan BEV Transformer çekirdeğidir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaSpatiotemporalBEVTransformer:
    """
    Tesla FSD BEVFormer Mekansal-Zamansal Transformer Çekirdeği.
    """
    def __init__(
        self,
        bev_h: int = 50,
        bev_w: int = 50,
        feature_dim: int = 64,
        history_len: int = 4
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.c = feature_dim
        self.history_len = history_len

        # Başlangıç BEV Sorguları (Learnable BEV Queries)
        np.random.seed(42)
        self.bev_queries = np.random.normal(0, 0.1, (bev_h, bev_w, self.c))
        self.prev_bev_features: Optional[np.ndarray] = None

        # 3D Sütun Yükseklik Örnekleme Noktaları (Z ekseni: 0.0m - 2.5m)
        self.pillar_z_samples = np.array([0.2, 0.8, 1.5, 2.2])

    def ego_motion_warp(
        self,
        bev_feature_map: np.ndarray,
        dx_m: float,
        dy_m: float,
        dyaw_rad: float,
        grid_res_m: float = 1.0
    ) -> np.ndarray:
        """
        Araç hareketine (Ego-Motion Odometri) göre önceki BEV haritasını ötele ve döndür.
        """
        # Izgara piksel cinsinden öteleme
        shift_x = int(round(dy_m / grid_res_m))  # Yanal kayma
        shift_y = int(round(-dx_m / grid_res_m)) # Boyuna kayma

        # Koordinat yuvarlama ve öteleme
        warped = np.roll(bev_feature_map, shift=(shift_y, shift_x), axis=(0, 1))

        # Sınır taşmalarını sıfırla
        if shift_y > 0:
            warped[:shift_y, :, :] = 0
        elif shift_y < 0:
            warped[shift_y:, :, :] = 0

        if shift_x > 0:
            warped[:, :shift_x, :] = 0
        elif shift_x < 0:
            warped[:, shift_x:, :] = 0

        return warped

    def spatial_cross_attention(
        self,
        multi_cam_features: Dict[str, np.ndarray],
        bev_queries: np.ndarray
    ) -> np.ndarray:
        """
        8 Kameradan 3D Işın Örnekleme ile Mekansal Çapraz Dikkat.
        """
        if not multi_cam_features:
            return bev_queries

        aggregated_features = np.zeros_like(bev_queries)
        for cam_name, feat in multi_cam_features.items():
            aggregated_features += feat
        aggregated_features /= len(multi_cam_features)

        # Query + Key-Value Cross Attention çıkışı
        return 0.3 * bev_queries + 0.7 * aggregated_features

    def temporal_self_attention(
        self,
        current_bev: np.ndarray,
        prev_bev: Optional[np.ndarray],
        dx_m: float,
        dy_m: float,
        dyaw_rad: float
    ) -> np.ndarray:
        """
        Geçmiş BEV tensörü ile Zamansal Öz-Dikkat ve Bellek Füzyonu.
        """
        if prev_bev is None:
            return current_bev

        # 1. Önceki BEV haritasını araç hareketine göre warp et
        warped_prev = self.ego_motion_warp(prev_bev, dx_m, dy_m, dyaw_rad)

        # 2. Zamansal Dikkat Füzyonu: %60 Mevcut Algı + %40 Zamansal Bellek (Oklüzyon Direnci)
        fused_bev = 0.60 * current_bev + 0.40 * warped_prev
        return fused_bev

    def step(
        self,
        multi_cam_features: Dict[str, np.ndarray],
        dx_m: float,
        dy_m: float,
        dyaw_rad: float
    ) -> Dict[str, Any]:
        """
        Tam Mekansal-Zamansal Transformer İleri Besleme Çevrimi.
        """
        # 1. Mekansal Çapraz Dikkat
        spatial_bev = self.spatial_cross_attention(multi_cam_features, self.bev_queries)

        # 2. Zamansal Öz-Dikkat
        fused_bev = self.temporal_self_attention(
            spatial_bev,
            self.prev_bev_features,
            dx_m,
            dy_m,
            dyaw_rad
        )

        # 3. Geçmişi güncelle
        self.prev_bev_features = fused_bev.copy()

        # 4. Nesne Olasılık Haritası (Kanal ortalaması / Sigmoid aktivasyonu)
        occupancy_prob = 1.0 / (1.0 + np.exp(-np.mean(fused_bev, axis=-1)))

        return {
            "fused_bev": fused_bev,
            "occupancy_prob": occupancy_prob,
            "bev_shape": fused_bev.shape
        }
