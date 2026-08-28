"""Day 25 Birim Testleri: Nesne Tespiti, IoU, NMS ve Anchor Box Regresyonu."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.kutu_donusturucu import KutuDonusturucu
from src.iou_hesaplayici import IoUHesaplayici
from src.nms_filtresi import NMSFiltresi
from src.anchor_ureteci import AnchorUreteci
from src.gorsellestirici import TespitGorsellestirici


def test_kutu_donusumleri_numpy_ve_torch():
    """Pascal VOC, COCO ve YOLO format dönüşümlerinin çift yönlü tutarlılığını test eder."""
    boxes_np = np.array([
        [10.0, 20.0, 50.0, 80.0],
        [100.0, 150.0, 200.0, 300.0],
    ])
    boxes_th = torch.from_numpy(boxes_np)

    # NumPy dönüşümleri
    xywh_np = KutuDonusturucu.xyxy_to_xywh(boxes_np)
    xyxy_rev_np = KutuDonusturucu.xywh_to_xyxy(xywh_np)
    np.testing.assert_allclose(boxes_np, xyxy_rev_np)

    cxcywh_np = KutuDonusturucu.xyxy_to_cxcywh(boxes_np)
    cxcywh_rev_np = KutuDonusturucu.cxcywh_to_xyxy(cxcywh_np)
    np.testing.assert_allclose(boxes_np, cxcywh_rev_np)

    # PyTorch dönüşümleri
    xywh_th = KutuDonusturucu.xyxy_to_xywh(boxes_th)
    xyxy_rev_th = KutuDonusturucu.xywh_to_xyxy(xywh_th)
    assert torch.allclose(boxes_th, xyxy_rev_th)


def test_kutu_normalizasyon_ve_kirpma():
    """Koordinat normalizasyonu ve sınır dışı kırpma işlemlerini test eder."""
    boxes = np.array([
        [-10.0, 50.0, 550.0, 600.0],
        [50.0, 60.0, 150.0, 200.0],
    ])
    img_w, img_h = 500, 500

    clipped = KutuDonusturucu.kirp_sinirla(boxes, img_w, img_h)
    assert clipped[0, 0] == 0.0
    assert clipped[0, 2] == 500.0
    assert clipped[0, 3] == 500.0

    norm = KutuDonusturucu.normalize_et(clipped, img_w, img_h)
    assert 0.0 <= np.min(norm) and np.max(norm) <= 1.0

    denorm = KutuDonusturucu.denormalize_et(norm, img_w, img_h)
    np.testing.assert_allclose(clipped, denorm)


def test_iou_ve_giou_hesaplama():
    """Tam çakışan, kısmi çakışan ve tamamen ayrık kutularda IoU ve GIoU değerlerini test eder."""
    b1 = np.array([[0.0, 0.0, 10.0, 10.0]])  # Alan 100
    b2 = np.array([[0.0, 0.0, 10.0, 10.0]])  # Birebir aynı
    b3 = np.array([[5.0, 0.0, 15.0, 10.0]])  # Yarı yarıya çakışan (Kesişim 50, Birleşim 150 -> IoU 1/3)
    b4 = np.array([[20.0, 20.0, 30.0, 30.0]]) # Tamamen ayrık

    # Tam çakışma
    assert IoUHesaplayici.iou_matrisi(b1, b2)[0, 0] == pytest.approx(1.0)
    assert IoUHesaplayici.giou_matrisi(b1, b2)[0, 0] == pytest.approx(1.0)

    # Kısmi çakışma
    assert IoUHesaplayici.iou_matrisi(b1, b3)[0, 0] == pytest.approx(50.0 / 150.0)

    # Ayrık kutular
    assert IoUHesaplayici.iou_matrisi(b1, b4)[0, 0] == 0.0
    # Ayrık kutularda GIoU negatif olmalıdır
    assert IoUHesaplayici.giou_matrisi(b1, b4)[0, 0] < 0.0


def test_klasik_nms():
    """NMS algoritmasının çakışan düşük skorlu kutuları elediğini test eder."""
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 51.0, 49.0],  # İlk kutuyla IoU > 0.8
        [200.0, 200.0, 250.0, 250.0], # Farklı nesne
    ])
    scores = np.array([0.90, 0.75, 0.85])

    secilenler = NMSFiltresi.klasik_nms(boxes, scores, iou_esigi=0.5, skor_esigi=0.3)
    assert len(secilenler) == 2
    assert 0 in secilenler
    assert 2 in secilenler
    assert 1 not in secilenler


def test_sinifa_duyarli_nms():
    """Farklı sınıflara ait çakışan kutuların birbirini ezmediğini test eder."""
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [10.0, 10.0, 50.0, 50.0],  # Aynı konumda farklı sınıf
    ])
    scores = np.array([0.90, 0.85])
    labels = np.array([0, 1])  # 0: Araba, 1: Bisiklet

    secilenler = NMSFiltresi.sinifa_duyarli_nms(boxes, scores, labels, iou_esigi=0.5)
    assert len(secilenler) == 2


def test_soft_nms():
    """Soft-NMS algoritmasının skorları sönümlediğini doğrular."""
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 51.0, 49.0],
    ])
    scores = np.array([0.90, 0.80])

    _, out_scores, _ = NMSFiltresi.soft_nms(boxes, scores, iou_esigi=0.5, method="gaussian")
    # İkinci kutunun skoru sönümlenmiş olmalı
    assert out_scores[1] < 0.80


def test_anchor_uretimi_ve_regresyon_cozumu():
    """Anchor grid üretimini, GT eşlemesini ve delta regresyon çözümünü test eder."""
    anchors = AnchorUreteci.anchor_grid_uret(grid_w=4, grid_h=4, stride=32, olcekler=(32.0,), en_boy_oranlari=(1.0,))
    assert len(anchors) == 16

    gt_boxes = np.array([
        [10.0, 10.0, 40.0, 40.0],
    ])

    etiketler, eslesen_gt, _ = AnchorUreteci.ground_truth_esle(anchors, gt_boxes, pos_iou_esigi=0.4, neg_iou_esigi=0.2)
    assert np.sum(etiketler == 1) >= 1

    # Regresyon tersinirliği testi
    poz_idx = np.where(etiketler == 1)[0]
    p_anchors = anchors[poz_idx]
    matched_gt = gt_boxes[eslesen_gt[poz_idx]]

    deltalar = AnchorUreteci.kutu_regresyon_hedefleri(p_anchors, matched_gt)
    cozulmus = AnchorUreteci.regresyondan_kutulari_coz(p_anchors, deltalar)

    np.testing.assert_allclose(cozulmus, matched_gt, atol=1e-5)


def test_dashboard_gorsellestirici(tmp_path):
    """4 panelli nesne tespiti görselinin oluşturulduğunu test eder."""
    gt_boxes = np.array([[50.0, 50.0, 150.0, 150.0]])
    gt_labels = ["Araba"]
    raw_boxes = np.array([[48.0, 52.0, 152.0, 148.0]])
    raw_scores = np.array([0.92])
    raw_labels = ["Araba"]
    nms_boxes = raw_boxes
    nms_scores = raw_scores
    nms_labels = raw_labels
    anchors = np.array([[40.0, 40.0, 160.0, 160.0]])
    iou_mat = np.array([[0.85]])
    giou_mat = np.array([[0.85]])

    hedef = tmp_path / "test_tespit_paneli.png"
    cikti = TespitGorsellestirici.dashboard_ciz(
        gt_boxes, gt_labels, raw_boxes, raw_scores, raw_labels,
        nms_boxes, nms_scores, nms_labels, anchors, iou_mat, giou_mat,
        img_size=(256, 256), hedef_dosya=hedef
    )

    assert cikti.exists()
    assert cikti.stat().st_size > 0
