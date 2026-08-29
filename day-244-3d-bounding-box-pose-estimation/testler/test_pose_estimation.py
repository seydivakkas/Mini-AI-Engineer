"""
PyTest Birim Testleri - Day 244: 3D Sınırlayıcı Kutu ve 6-DoF Duruş Kestirimi (VoteNet) Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose_estimation_motoru import (
    VotingModule,
    BoundingBox3DHead,
    VoteNetPoseEstimator,
    hesapla_adds_metrigi,
)
from src.pose_profilleyici import PoseEstimationProfilleyici
from src.gorsellestirici import PoseEstimationGorsellestirici


def test_voting_module_output_shape():
    """1. VotingModule doğru oy koordinatları ve özellikleri üretmelidir."""
    voter = VotingModule(in_channels=32, out_channels=32)
    xyz = torch.randn(2, 64, 3)
    feat = torch.randn(2, 32, 64)
    v_xyz, v_feat = voter(xyz, feat)
    assert v_xyz.shape == (2, 64, 3)
    assert v_feat.shape == (2, 32, 64)


def test_bounding_box_head_keys():
    """2. BoundingBox3DHead tüm 3D kutu parametrelerini eksiksiz döndürmelidir."""
    head = BoundingBox3DHead(in_channels=32)
    cluster = torch.randn(2, 32)
    res = head(cluster)
    assert "center" in res
    assert "dimensions" in res
    assert "yaw_rad" in res
    assert "confidence" in res


def test_bounding_box_positive_dimensions():
    """3. Kutu boyutları (l, w, h) kesinlikle pozitif olmalıdır."""
    head = BoundingBox3DHead(in_channels=32)
    cluster = torch.randn(4, 32)
    res = head(cluster)
    assert torch.all(res["dimensions"] > 0)


def test_votenet_pose_estimator_forward():
    """4. VoteNetPoseEstimator [B, N, 3] girdisini işleyip tahmin üretmelidir."""
    model = VoteNetPoseEstimator(feature_dim=32)
    xyz = torch.randn(2, 128, 3)
    res = model(xyz)
    assert res["center"].shape == (2, 3)
    assert res["dimensions"].shape == (2, 3)


def test_adds_metric_success_threshold():
    """5. hesapla_adds_metrigi mesafe ve başarı bayrağını doğru hesaplamalıdır."""
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([0.01, 0.0, 0.0])  # 1 cm
    hata, basarili = hesapla_adds_metrigi(c1, c2, esik_cm=2.0)
    assert hata == 1.0
    assert basarili is True


def test_profiler_pose_metrics():
    """6. Profilleyici 3D mAP değerinin %80 üstünde olduğunu doğrulamalıdır."""
    prof = PoseEstimationProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["3d_map_0_5_skoru"]["VoteNet_6DoF"]
    assert skor > 80.0
    assert prof["karsilastirma"]["adds_2cm_tutma_dogrulugu"]["VoteNet_6DoF"] > 85.0


def test_yaw_angle_bounds():
    """7. Yaw açısı [-pi, pi] sınırları içinde olmalıdır."""
    head = BoundingBox3DHead(in_channels=32)
    cluster = torch.randn(5, 32)
    res = head(cluster)
    assert torch.all(res["yaw_rad"] >= -np.pi) and torch.all(res["yaw_rad"] <= np.pi)


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PoseEstimationGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_pose_paneli.png")
    profil = PoseEstimationProfilleyici.basarim_profili_cikar()

    PoseEstimationGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
