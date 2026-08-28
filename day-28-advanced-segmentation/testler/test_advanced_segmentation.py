"""
Day 28: İleri Düzey Bölütleme, Mask R-CNN, SegFormer ve Panoptik Metrikleri Birim Testleri.
"""

import os
import numpy as np
import pytest
import torch

from src.bolutleme_turleri import BolutlemeTipi, PanoptikDonusturucu
from src.mask_rcnn_modulu import MaskRCNNYoneticisi, RoIAlignModulu, FCNMaskeBasligi
from src.segformer_mimari import SegFormerModeli
from src.panoptik_ve_mask_metrikleri import PanoptikMetrikHesaplayici, MaskeIoUHesaplayici
from src.sentetik_sahne_ureteci import SentetikSahneUreteci
from src.gorsellestirici import IleriBolutlemeGorsellestirici


def test_bolutleme_turleri_ve_panoptik_donusturucu():
    """Panoptik harita birleştirme ve ID kodlama/çözme testi."""
    semantik = np.zeros((100, 100), dtype=np.int64)
    semantik[:50, :] = 0  # Gökyüzü (Stuff)
    semantik[50:, :] = 1  # Yol (Stuff)

    # 1 adet araç maskesi
    mask1 = np.zeros((100, 100), dtype=np.float32)
    mask1[60:80, 20:60] = 1.0
    semantik[mask1 > 0] = 2  # Araç

    panoptik, segmentler = PanoptikDonusturucu.birlestir_panoptik(
        semantik_harita=semantik,
        ornek_maskeleri=[mask1],
        ornek_siniflari=[2]
    )

    assert panoptik.shape == (100, 100)
    assert len(segmentler) >= 2
    # ID çözme testi
    sinif_id, ornek_id = PanoptikDonusturucu.id_coz(2001)
    assert sinif_id == 2
    assert ornek_id == 1


def test_mask_rcnn_roialign_ve_mask_head():
    """RoIAlign ve FCN Maske Başlığı tensör boyutları testi."""
    feature_map = torch.randn(2, 256, 64, 64)
    rois = torch.tensor([
        [0, 10.0, 10.0, 30.0, 30.0],
        [1, 15.0, 20.0, 45.0, 50.0]
    ], dtype=torch.float32)

    roi_align = RoIAlignModulu(cikti_boyutu=(14, 14), spatial_scale=1.0, ornekleme_orani=2)
    roi_feats = roi_align(feature_map, rois)
    assert roi_feats.shape == (2, 256, 14, 14)

    mask_head = FCNMaskeBasligi(girdi_kanali=256, sinif_sayisi=5)
    mask_logits = mask_head(roi_feats)
    assert mask_logits.shape == (2, 5, 28, 28)


def test_mask_rcnn_coklu_gorev_kaybi():
    """Çok Görevli Kayıp (Multi-Task Loss: L_cls + L_box + L_mask) testi."""
    cls_logits = torch.randn(2, 4, requires_grad=True)
    box_deltas = torch.randn(2, 16, requires_grad=True)
    mask_logits = torch.randn(2, 4, 28, 28, requires_grad=True)

    gt_labels = torch.tensor([1, 2], dtype=torch.long)
    gt_boxes = torch.randn(2, 4)
    gt_masks = torch.randint(0, 2, (2, 28, 28)).float()

    kayiplar = MaskRCNNYoneticisi.coklu_gorev_kaybi(
        cls_logits, box_deltas, mask_logits, gt_labels, gt_boxes, gt_masks
    )

    assert "toplam_kayip" in kayiplar
    assert kayiplar["toplam_kayip"].item() > 0
    # Geri yayılım testi
    kayiplar["toplam_kayip"].backward()
    assert cls_logits.grad is not None
    assert mask_logits.grad is not None


def test_segformer_transformer_ileri_besleme():
    """SegFormer çok ölçekli transformer ileri besleme testi."""
    model = SegFormerModeli(in_channels=3, num_classes=4, embed_dims=[16, 32, 64, 128])
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 4, 64, 64)


def test_panoptik_kalite_ve_mask_iou():
    """Maske IoU ve Panoptik Kalite (PQ = SQ x RQ) metriği testi."""
    m1 = np.zeros((50, 50), dtype=np.float32)
    m1[10:30, 10:30] = 1.0
    m2 = np.zeros((50, 50), dtype=np.float32)
    m2[10:30, 10:30] = 1.0

    # Kusursuz eşleşmede IoU = 1.0
    assert pytest.approx(MaskeIoUHesaplayici.iou(m1, m2), 0.001) == 1.0

    # Panoptik Kalite testi (Birebir eşleşen iki harita)
    gt_panoptic = np.zeros((50, 50), dtype=np.int32)
    gt_panoptic[:25, :] = 1000  # Stuff 1
    gt_panoptic[25:, :] = 2001  # Thing 2

    pq_res = PanoptikMetrikHesaplayici.hesapla_pq(gt_panoptic, gt_panoptic, iou_esigi=0.5)
    assert pytest.approx(pq_res["genel_pq"], 0.001) == 1.0
    assert pytest.approx(pq_res["genel_sq"], 0.001) == 1.0
    assert pytest.approx(pq_res["genel_rq"], 0.001) == 1.0


def test_sentetik_sahne_ureteci_ve_gorsellestirici(tmp_path):
    """Sahne üretimi ve 6 panelli teşhis panosu çizim testi."""
    uretec = SentetikSahneUreteci(img_size=128, seed=42)
    sahne = uretec.sahne_uret(num_instances=3)

    assert sahne["gorsel_rgb"].shape == (128, 128, 3)
    assert len(sahne["ornek_maskeleri"]) == 3

    roi_ornek = np.random.rand(28, 28)
    mask_pred = np.random.rand(28, 28)
    metrikler = {"genel_pq": 0.88, "genel_sq": 0.92, "genel_rq": 0.95, "AP_50": 0.90, "AP_75": 0.80}

    out_file = str(tmp_path / "test_teshis.png")
    cizim_path = IleriBolutlemeGorsellestirici.teshis_panosu_ciz(
        sahne_verisi=sahne,
        roi_align_ornek=roi_ornek,
        mask_pred_ornek=mask_pred,
        metrikler=metrikler,
        hedef_path=out_file
    )
    assert os.path.exists(cizim_path)
