"""
PyTest Birim Testleri - Day 258: Sıfır Örnekli (Zero-Shot) Görülmemiş Nesneleri Kavrama ve Ayırma.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.zero_shot_grasping_motoru import (
    PointCloudPreprocessor,
    AntipodalGraspGenerator,
    ZeroShotBinSortingPipeline,
)
from src.zero_shot_grasping_profilleyici import ZeroShotGraspingProfilleyici
from src.gorsellestirici import ZeroShotGraspingGorsellestirici


def test_point_cloud_filter_table_plane():
    """1. filter_table_plane z < table_z_min noktalarını elemelidir."""
    pts = np.array([[0, 0, 0.01], [0, 0, 0.05], [0, 0, 0.10]])
    filtered = PointCloudPreprocessor.filter_table_plane(pts, table_z_min=0.02)
    assert len(filtered) == 2
    assert np.all(filtered[:, 2] >= 0.02)


def test_point_cloud_estimate_normals():
    """2. estimate_normals birim uzunluklu yüzey normalleri üretmelidir."""
    pts = np.random.randn(20, 3)
    normals = PointCloudPreprocessor.estimate_normals(pts, k_neighbors=5)
    assert normals.shape == (20, 3)
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_antipodal_grasp_quality_valid():
    """3. evaluate_grasp_quality birbirine bakan zıt normallerde pozitif kalite dönmelidir."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.05, 0.0, 0.0])
    n1 = np.array([-1.0, 0.0, 0.0])
    n2 = np.array([1.0, 0.0, 0.0])
    q = AntipodalGraspGenerator.evaluate_grasp_quality(p1, p2, n1, n2)
    assert q > 0.5


def test_antipodal_grasp_quality_invalid():
    """4. evaluate_grasp_quality aynı yöne bakan normallerde 0.0 kalite dönmelidir."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.05, 0.0, 0.0])
    n1 = np.array([1.0, 0.0, 0.0])
    n2 = np.array([1.0, 0.0, 0.0])
    q = AntipodalGraspGenerator.evaluate_grasp_quality(p1, p2, n1, n2)
    assert q == 0.0


def test_generate_grasps_candidates():
    """5. generate_grasps nokta bulutundan kaliteli tutuş pozları üretmelidir."""
    # Küre / silindir benzeri simetrik nokta çiftleri
    p1 = np.array([0.0, 0.0, 0.05])
    p2 = np.array([0.04, 0.0, 0.05])
    pts = np.array([p1, p2, [0.02, 0.02, 0.05], [0.02, -0.02, 0.05]])
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    grasps = AntipodalGraspGenerator.generate_grasps(pts, normals, max_grasps=5)
    assert len(grasps) >= 1


def test_zero_shot_bin_sorting_pipeline():
    """6. sort_unseen_object görülmemiş nesneyi hedef kutuya başarıyla yönlendirmelidir."""
    pts = np.random.randn(30, 3) * 0.02 + np.array([0.5, 0.0, 0.05])
    res = ZeroShotBinSortingPipeline.sort_unseen_object(pts, semantic_category="ORGANİK")
    assert res["durum"] == "BASARILI_AYRILDI"
    assert res["hedef_kutu_koordinati"] == [0.60, 0.35, 0.20]
    assert "secilen_6dof_grasp" in res


def test_zero_shot_grasping_profiler_output():
    """7. ZeroShotGraspingProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = ZeroShotGraspingProfilleyici.basarim_profili_cikar()
    assert "Zero_Shot_AnyGrasp" in profil["karsilastirma"]["gorulmemis_nesne_kavrama_basarisi_yuzde"]
    assert profil["karsilastirma"]["gorulmemis_nesne_kavrama_basarisi_yuzde"]["Zero_Shot_AnyGrasp"] == 97.6


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ZeroShotGraspingGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_zero_shot_grasping_paneli.png")
    profil = ZeroShotGraspingProfilleyici.basarim_profili_cikar()

    ZeroShotGraspingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
