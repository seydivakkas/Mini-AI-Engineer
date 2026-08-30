"""
Day 333: Neuromorphic Spatial Navigation & Grid/Place Cells
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Entorhinal Korteks 60-Derece Hekzagonal Grid Hücrelerini, Hipokampal Konum (Place) Hücrelerini
ve Yol Entegrasyonu (Path Integration) Tabanlı Nöromorfik Navigasyon Motorunu içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class GridCellModule:
    """
    Entorhinal Korteks Hekzagonal Grid (Izgara) Hücresi Modülü.
    2D Uzayda 60 derecelik periyodik hekzagonal uyarım haritası oluşturur.
    """
    def __init__(self, spatial_scale: float = 1.5, phase_offset: Tuple[float, float] = (0.0, 0.0)):
        self.spatial_scale = spatial_scale
        self.phase_offset = np.array(phase_offset, dtype=np.float32)

        # 60 Derece Hekzagonal Dalga Vektörleri
        angles = [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0]
        self.wave_vectors = [np.array([np.cos(a), np.sin(a)], dtype=np.float32) for a in angles]

    def compute_firing_rate(self, position: np.ndarray) -> float:
        """
        Girdi: 2D Konum (x, y) -> Çıktı: Grid Hücresi Ateşleme Frekansı [0, 1]
        """
        rel_pos = position - self.phase_offset
        freq_factor = (4.0 * np.pi) / (math.sqrt(3.0) * self.spatial_scale)

        cosine_sum = 0.0
        for k in self.wave_vectors:
            cosine_sum += np.cos(freq_factor * np.dot(k, rel_pos))

        firing_rate = (2.0 / 3.0) * (cosine_sum / 3.0 + 0.5)
        return float(np.clip(firing_rate, 0.0, 1.0))


class PlaceCellNetwork:
    """
    Hipokampal Konum (Place) Hücreleri Ağı.
    Spesifik 2D lokal Gaussian duyusal alanlara (Place Fields) sahiptir.
    """
    def __init__(self, grid_size: int = 5, field_sigma: float = 0.4, env_bounds: float = 3.0):
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.field_sigma = field_sigma

        # Nöronların merkez konumları (Place field centers)
        coords = np.linspace(-env_bounds / 2.0, env_bounds / 2.0, grid_size)
        self.place_centers = []
        for x in coords:
            for y in coords:
                self.place_centers.append(np.array([x, y], dtype=np.float32))
        self.place_centers = np.array(self.place_centers, dtype=np.float32)

    def compute_place_rates(self, position: np.ndarray) -> np.ndarray:
        """
        Girdi: 2D Konum (x,y) -> Çıktı: (Num_Place_Cells,) Ateşleme Oranları
        """
        dists_sq = np.sum((self.place_centers - position) ** 2, axis=1)
        rates = np.exp(-dists_sq / (2.0 * (self.field_sigma ** 2)))
        return rates.astype(np.float32)

    def decode_position(self, rates: np.ndarray) -> np.ndarray:
        """
        Popülasyon Ağırlık Merkezi (Center of Mass) ile 2D Konum Kod Çözümü
        """
        weights = rates / (np.sum(rates) + 1e-9)
        decoded_pos = np.sum(self.place_centers * weights[:, np.newaxis], axis=0)
        return decoded_pos.astype(np.float32)


class NeuromorphicSpatialNavigator:
    """
    Nöromorfik Yol Entegrasyonu ve Otonom Mekansal Navigasyon Motoru.
    Girdi: Hız Vektörü (v_x, v_y) -> Çıktı: Grid/Place Aktivasyonu ve Tahmini Konum
    """
    def __init__(self, initial_position: Tuple[float, float] = (0.0, 0.0)):
        self.true_position = np.array(initial_position, dtype=np.float32)
        self.estimated_position = np.array(initial_position, dtype=np.float32)

        # 4 Farklı Ölçekte Grid Modülü
        self.grid_modules = [
            GridCellModule(spatial_scale=1.0, phase_offset=(0.0, 0.0)),
            GridCellModule(spatial_scale=1.5, phase_offset=(0.2, -0.1)),
            GridCellModule(spatial_scale=2.0, phase_offset=(-0.3, 0.3)),
            GridCellModule(spatial_scale=2.5, phase_offset=(0.1, 0.2)),
        ]
        self.place_network = PlaceCellNetwork(grid_size=5, field_sigma=0.5, env_bounds=4.0)

    def update_navigation_step(self, velocity: np.ndarray, dt: float = 0.1) -> Dict[str, Any]:
        """
        Hız İtme Adımı ve Nöronal Temsil Güncellemesi
        """
        # Gerçek ve Tahmini Konum Güncellemesi (Path Integration)
        self.true_position += velocity * dt
        self.estimated_position += velocity * dt

        # Grid ve Place Hücre Aktivasyonları
        grid_rates = [g.compute_firing_rate(self.estimated_position) for g in self.grid_modules]
        place_rates = self.place_network.compute_place_rates(self.estimated_position)
        decoded_pos = self.place_network.decode_position(place_rates)

        error_meters = float(np.linalg.norm(self.true_position - decoded_pos))

        return {
            "true_pos": self.true_position.copy(),
            "estimated_pos": self.estimated_position.copy(),
            "decoded_pos": decoded_pos.copy(),
            "grid_rates": grid_rates,
            "place_rates": place_rates,
            "error_meters": error_meters,
        }
