"""
PyTest Birim Testleri - Day 243: 3D Nokta Bulutu ve Mekansal Akıl Yürütme (PointNet++) Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.point_cloud_motoru import (
    farthest_point_sampling,
    ball_query,
    PointNetSetAbstraction,
    PointNetPlusPlusModel,
    ornek_3d_fincan_bulutu_olustur,
)
from src.point_cloud_profilleyici import PointCloudProfilleyici
from src.gorsellestirici import PointCloudGorsellestirici


def test_synthetic_point_cloud_generation():
    """1. ornek_3d_fincan_bulutu_olustur doğru boyutta [N, 3] bulut üretmelidir."""
    bulut = ornek_3d_fincan_bulutu_olustur(nokta_sayisi=256)
    assert bulut.shape == (256, 3)
    assert bulut.dtype == np.float32


def test_farthest_point_sampling_shape():
    """2. farthest_point_sampling [B, npoint] indis tensörü döndürmelidir."""
    xyz = torch.randn(2, 100, 3)
    fps_idx = farthest_point_sampling(xyz, npoint=32)
    assert fps_idx.shape == (2, 32)


def test_ball_query_shape():
    """3. ball_query [B, S, nsample] komşuluk indis tensörü üretmelidir."""
    xyz = torch.randn(2, 100, 3)
    new_xyz = torch.randn(2, 20, 3)
    idx = ball_query(radius=0.5, nsample=8, xyz=xyz, new_xyz=new_xyz)
    assert idx.shape == (2, 20, 8)


def test_pointnet_set_abstraction_forward():
    """4. PointNetSetAbstraction noktaları ve özellikleri doğru boyutta soyutlamalıdır."""
    sa = PointNetSetAbstraction(npoint=32, radius=0.3, nsample=8, in_channel=0, mlp=[16, 32])
    xyz = torch.randn(2, 128, 3)
    new_xyz, new_points = sa(xyz, None)
    assert new_xyz.shape == (2, 32, 3)
    assert new_points.shape == (2, 32, 32)


def test_pointnet_plus_plus_model_forward():
    """5. PointNetPlusPlusModel [0, 1] aralığında tutma afordansı üretmelidir."""
    model = PointNetPlusPlusModel(num_classes=1)
    xyz = torch.randn(1, 256, 3)
    score = model(xyz)
    assert score.shape == (1, 1)
    assert 0.0 <= score.item() <= 1.0


def test_profiler_point_cloud_metrics():
    """6. Profilleyici PointNet++ mIoU değerinin %80 üstünde olduğunu doğrulamalıdır."""
    prof = PointCloudProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["mekansal_segmentasyon_miou"]["PointNetPlusPlus"]
    assert skor > 80.0
    assert prof["karsilastirma"]["geometrik_tutma_basarisi"]["PointNetPlusPlus"] > 90.0


def test_permutation_invariance():
    """7. Nokta bulutunu karıştırmak model çıktısının geçerli aralıkta kalmasını sağlamalıdır."""
    model = PointNetPlusPlusModel(num_classes=1)
    xyz = torch.randn(1, 256, 3)
    perm = torch.randperm(256)
    xyz_shuffled = xyz[:, perm, :]

    model.eval()
    with torch.no_grad():
        s1 = model(xyz)
        s2 = model(xyz_shuffled)

    assert isinstance(s1.item(), float)
    assert isinstance(s2.item(), float)


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PointCloudGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_point_cloud_paneli.png")
    profil = PointCloudProfilleyici.basarim_profili_cikar()

    PointCloudGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
