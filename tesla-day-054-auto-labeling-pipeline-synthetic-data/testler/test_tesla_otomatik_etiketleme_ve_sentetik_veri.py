"""
Tesla Otomatik Etiketleme Birim Testleri (PyTest)
=================================================
Bu test paketi; Çift yönlü zamansal düzeltmeyi, çoklu sürüş harita hizalamasını,
3D IoU hesaplamasını ve sentetik sahne varyasyonlarını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_otomatik_etiketleme_ve_sentetik_veri import TeslaAutoLabelingPipeline


def test_cift_yonlu_zamansal_duzeltme_gurultu_azalmasi():
    """Çift yönlü filtrenin rastgele gürültüyü azalttığı (RMSE_smooth < RMSE_noisy) test edilir."""
    pipeline = TeslaAutoLabelingPipeline()
    t = np.linspace(0, 5, 50)
    clean_traj = np.column_stack([t, t * 2.0])
    np.random.seed(42)
    noisy_traj = clean_traj + np.random.normal(0, 0.2, clean_traj.shape)

    smoothed = pipeline.bidirectional_temporal_smoothing(noisy_traj)

    rmse_noisy = np.sqrt(np.mean((noisy_traj - clean_traj) ** 2))
    rmse_smooth = np.sqrt(np.mean((smoothed - clean_traj) ** 2))

    assert rmse_smooth < rmse_noisy


def test_coklu_surus_nokta_birlestirme():
    """İki ayrı sürüşten gelen nokta bulutlarının başarıyla birleştirildiği test edilir."""
    pipeline = TeslaAutoLabelingPipeline()
    p1 = np.ones((50, 3))
    p2 = np.ones((50, 3)) * 2.0

    res = pipeline.align_multi_trip_point_clouds(p1, p2)

    assert res["total_points"] == 100
    assert res["alignment_rmse_cm"] < 5.0


def test_3d_bbox_iou_kalite_dogrulamasi():
    """Birebir aynı kutuların IoU değerinin 1.0, hafif ötelenmiş kutuların > 0.85 olduğu test edilir."""
    pipeline = TeslaAutoLabelingPipeline()
    box1 = np.array([0.0, 10.0, 0.0, 2.0, 4.0, 1.5])
    box2 = np.array([0.0, 10.0, 0.0, 2.0, 4.0, 1.5])
    box3 = np.array([0.05, 10.05, 0.02, 2.0, 4.0, 1.5])

    iou_exact = pipeline.calculate_3d_bbox_iou(box1, box2)
    iou_shifted = pipeline.calculate_3d_bbox_iou(box1, box3)

    assert np.isclose(iou_exact, 1.0)
    assert iou_shifted > 0.85


def test_sentetik_hava_durumu_varyasyonlari():
    """Sentetik yağmur ve sis modlarının orijinal görüntüyü modifiye ettiği test edilir."""
    pipeline = TeslaAutoLabelingPipeline()
    img = np.full((50, 50, 3), 100, dtype=np.uint8)

    rain_img = pipeline.generate_synthetic_weather_variants(img, variant_type="RAIN")
    night_img = pipeline.generate_synthetic_weather_variants(img, variant_type="NIGHT")

    assert not np.array_equal(rain_img, img)
    assert np.mean(night_img) < np.mean(img)
