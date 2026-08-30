"""
Day 351: Satellite Constellation Edge AI for Real-Time Wildfire & Thermal Anomaly Detection
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.satellite_wildfire_edge_motoru import (
    MultispectralEarthSimulator,
    OnBoardWildfireEdgeDetector,
    SatelliteConstellationNetwork,
)
from src.wildfire_profilleyici import WildfireProfilleyici


def test_multispectral_earth_simulator_dimensions():
    """
    Çok Bantlı Dünya Simülatörü Boyut Testi.
    """
    sim = MultispectralEarthSimulator(grid_size=32)
    img, mask = sim.generate_multispectral_tile(has_wildfire=True)
    
    assert img.shape == (4, 32, 32)
    assert mask.shape == (32, 32)
    assert np.sum(mask) > 0


def test_onboard_wildfire_edge_detector():
    """
    Uydu Üzeri Edge AI Yangın Algılama Testi.
    """
    sim = MultispectralEarthSimulator(grid_size=32)
    img, mask_true = sim.generate_multispectral_tile(has_wildfire=True)
    
    detector = OnBoardWildfireEdgeDetector()
    res = detector.detect_wildfire(img)
    
    assert "pred_mask" in res
    assert "alert_payload" in res
    assert res["alert_payload"]["satellite_edge_alert"] is True
    assert res["alert_payload"]["total_frp_mw"] > 0.0


def test_satellite_constellation_latency():
    """
    Küp Uydu Takımyıldızı İletim Gecikmesi Testi.
    """
    constellation = SatelliteConstellationNetwork(num_sats=6)
    lat = constellation.route_alert_to_ground({"alert": True})
    assert lat < 50.0 # 50 ms altında gecikme


def test_wildfire_profiler_metrics():
    """
    Uydu Yangın Profilleyici Testi.
    """
    m_true = np.zeros((10, 10), dtype=bool)
    m_true[2:5, 2:5] = True
    m_pred = m_true.copy()
    
    metrics = WildfireProfilleyici.profille(m_true, m_pred, total_frp_mw=45.0)
    assert metrics["recall_score"] == 100.0
    assert metrics["precision_score"] == 100.0
    assert metrics["iou_score"] == 1.0
