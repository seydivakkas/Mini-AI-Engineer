"""
3D Gaussian Splatting (3DGS) Test Paketi (Day 179 - FAZ 9).
8 adet kapsamlı PyTest birim testi.
"""

import sys
import os
import tempfile
import pytest
import torch
import numpy as np

# Proje dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gaussian_temsili import Gaussian3D, kuaterniyon_to_rotasyon_matrisi
from src.kovaryans_projeksiyonu import KovaryansProjeksiyonu
from src.diferansiyellenebilir_rasterizer import GaussianRasterizer
from src.gorsellestirici import GaussianGorsellestirici


def test_kuaterniyon_to_rotasyon_matrisi_orthogonality():
    """1. Kuaterniyon normalizasyonu ve R * R^T = I ortogonalite doğrulaması."""
    # Rastgele kuaterniyonlar [N, 4]
    q = torch.randn(10, 4)
    R = kuaterniyon_to_rotasyon_matrisi(q)

    assert R.shape == (10, 3, 3), "Rotasyon matrisi boyutu [N, 3, 3] olmalıdır."

    # R * R^T = I kontrolü
    I_approx = torch.bmm(R, R.transpose(1, 2))
    I_expected = torch.eye(3).unsqueeze(0).expand(10, 3, 3)

    assert torch.allclose(I_approx, I_expected, atol=1e-5), "Rotasyon matrisi ortogonal olmalıdır (R * R^T = I)."

    # Determinant kontrolü (det(R) == 1)
    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(10), atol=1e-5), "Rotasyon matrisinin determinantı +1 olmalıdır."


def test_gaussian3d_kovaryans_pozitif_yari_tanimli():
    """2. 3D Kovaryans matrisi Sigma = R * S * S^T * R^T simetriklik ve pozitif yarı-tanımlılık testi."""
    gaussians = Gaussian3D(num_gaussians=25)
    Sigma = gaussians.kovaryans_3d_hesapla()

    assert Sigma.shape == (25, 3, 3), "Kovaryans matrisi boyutu [N, 3, 3] olmalıdır."

    # 1. Simetriklik kontrolü: Sigma == Sigma^T
    Sigma_T = Sigma.transpose(1, 2)
    assert torch.allclose(Sigma, Sigma_T, atol=1e-5), "Kovaryans matrisi simetrik olmalıdır."

    # 2. Pozitif yarı-tanımlılık: Tüm özdeğerler >= 0
    eigenvalues = torch.linalg.eigvalsh(Sigma)
    assert (eigenvalues >= -1e-6).all(), "Kovaryans matrisinin tüm özdeğerleri pozitif olmalıdır."


def test_jacobian_hesaplama():
    """3. Kamera izdüşüm Jacobian matrisi J [N, 2, 3] ve türev doğrulaması."""
    pts_cam = torch.tensor([
        [0.0, 0.0, 2.0],
        [1.0, -1.0, 4.0],
        [0.5, 0.5, 0.05],  # Derinlik min 0.1 ile sınırlandırılmalı
    ], dtype=torch.float32)

    fx, fy = 500.0, 500.0
    J = KovaryansProjeksiyonu.jacobian_hesapla(pts_cam, fx=fx, fy=fy)

    assert J.shape == (3, 2, 3), "Jacobian matrisi boyutu [N, 2, 3] olmalıdır."

    # İlk nokta için (x=0, y=0, z=2): J[0, 0, 0] = fx/z = 250, J[0, 0, 2] = -fx*x/z^2 = 0
    assert pytest.approx(J[0, 0, 0].item(), rel=1e-4) == 250.0
    assert pytest.approx(J[0, 0, 2].item(), abs=1e-5) == 0.0
    assert pytest.approx(J[0, 1, 1].item(), rel=1e-4) == 250.0


def test_izdusum_2d_kovaryans_ve_ekran_merkezi():
    """4. 3D'den 2D ekrana kovaryans ve merkez projeksiyonu testi."""
    mu_3d = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
    sigma_3d = torch.eye(3).unsqueeze(0).expand(2, 3, 3) * 0.1
    R_cam = torch.eye(3)
    T_cam = torch.tensor([0.0, 0.0, 2.0])

    mu_2d, sigma_2d = KovaryansProjeksiyonu.izdusum_2d_kovaryans(
        mu_3d, sigma_3d, R_cam, T_cam, fx=200.0, fy=200.0, cx=32.0, cy=32.0
    )

    assert mu_2d.shape == (2, 2), "2D merkez boyutu [N, 2] olmalıdır."
    assert sigma_2d.shape == (2, 2, 2), "2D kovaryans boyutu [N, 2, 2] olmalıdır."

    # İlk nokta tam merkezde (cx=32, cy=32) olmalıdır
    assert pytest.approx(mu_2d[0, 0].item(), abs=1e-4) == 32.0
    assert pytest.approx(mu_2d[0, 1].item(), abs=1e-4) == 32.0

    # Anti-aliasing filtresi: Köşegenler en az +0.3 olmalıdır
    assert (sigma_2d[:, 0, 0] >= 0.3).all()
    assert (sigma_2d[:, 1, 1] >= 0.3).all()


def test_rasterizer_bos_ve_gecersiz_derinlik():
    """5. Kamera arkasındaki noktaların filtrelenmesi ve boş render durumu."""
    gaussians = Gaussian3D(num_gaussians=5)
    # Tüm Gaussları kamera arkasına (z = -5.0) taşı
    gaussians.mu.data[:, 2] = -5.0

    rasterizer = GaussianRasterizer(width=32, height=32)
    R_cam = torch.eye(3)
    T_cam = torch.tensor([0.0, 0.0, 0.0])

    out = rasterizer.render(gaussians, R_cam, T_cam)

    assert out["num_rendered"] == 0, "Kamera arkasındaki noktalar elenmelidir."
    assert out["image"].shape == (32, 32, 3), "Çıktı boyutu [H, W, 3] olmalıdır."
    assert torch.allclose(out["image"], torch.zeros(32, 32, 3)), "Görüntü siyah olmalıdır."


def test_rasterizer_render_ve_alfa_karistirma():
    """6. Diferansiyellenebilir alfa karıştırma ve gradyan akışı testi."""
    gaussians = Gaussian3D(num_gaussians=10)
    rasterizer = GaussianRasterizer(width=32, height=32)

    R_cam = torch.eye(3)
    T_cam = torch.tensor([0.0, 0.0, 3.0])

    out = rasterizer.render(gaussians, R_cam, T_cam)
    img = out["image"]

    assert img.shape == (32, 32, 3), "Render edilmiş görüntü [32, 32, 3] olmalıdır."
    assert (img >= 0.0).all() and (img <= 1.0).all(), "Piksel renkleri [0, 1] aralığında olmalıdır."

    # Gradyan akışı testi (Diferansiyellenebilirlik)
    loss = img.sum()
    loss.backward()
    assert gaussians.mu.grad is not None, "Gauss merkezleri mu için gradyan hesaplanabilmelidir."
    assert gaussians.log_scaling.grad is not None, "Gauss ölçekleri için gradyan hesaplanabilmelidir."


def test_kiyaslama_raporu_metrikleri():
    """7. NeRF vs 3DGS kıyaslama raporu doğrulaması."""
    rapor = GaussianRasterizer.ornek_3dgs_kiyaslama_raporu()

    assert "karsilastirma" in rapor, "Raporda karsilastirma anahtarı bulunmalıdır."
    assert len(rapor["karsilastirma"]) == 3, "3 yöntem kıyaslanmalıdır."

    yontem_adlari = [item["yontem"] for item in rapor["karsilastirma"]]
    assert any("3D Gaussian Splatting" in y for y in yontem_adlari)

    # 3DGS FPS değeri >= 100 olmalıdır
    gs_item = [item for item in rapor["karsilastirma"] if "3D Gaussian Splatting" in item["yontem"]][0]
    assert gs_item["fps"] >= 100.0, "3DGS gerçek zamanlı (>= 100 FPS) olmalıdır."


def test_gorsellestirme_cikti_dosyasi():
    """8. 6 panelli teşhis panosunun dosyaya eksiksiz kaydedilmesi testi."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kayit_yolu = os.path.join(tmpdir, "test_3dgs_paneli.png")
        rapor = GaussianRasterizer.ornek_3dgs_kiyaslama_raporu()

        gorsellestirici = GaussianGorsellestirici(dpi=100)
        gorsellestirici.pano_olustur(rapor=rapor, kayit_yolu=kayit_yolu)

        assert os.path.exists(kayit_yolu), "Görselleştirme dosyası kaydedilmiş olmalıdır."
        assert os.path.getsize(kayit_yolu) > 1000, "Dosya boyutu geçerli olmalıdır."
