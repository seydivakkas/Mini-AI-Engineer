"""Day 26 Birim Testleri: YOLO Eğitimi, mAP Hesaplayıcı ve Çıkarım."""

from pathlib import Path
import numpy as np
import pytest
import yaml

from src.sentetik_veri_ureteci import YOLOVeriUreteci
from src.map_hesaplayici import MAPHesaplayici
from src.yolo_yoneticisi import YOLOYoneticisi
from src.gorsellestirici import YOLOGorsellestirici


def test_yolo_veri_seti_ve_yaml(tmp_path):
    """Sentetik veri setinin ve data.yaml yapılandırmasının doğruluğunu test eder."""
    yaml_path = YOLOVeriUreteci.veri_seti_olustur(
        ana_dizin=tmp_path, train_adet=4, val_adet=2, img_w=256, img_h=256
    )

    assert yaml_path.exists()

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "names" in data
    assert len(data["names"]) == 3

    train_imgs = list((tmp_path / "images" / "train").glob("*.jpg"))
    train_lbls = list((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(train_imgs) == 4
    assert len(train_lbls) == 4


def test_iou_ve_sinif_ap_hesaplama():
    """AP hesaplayıcının mükemmel tahminlerde AP=1.0 ürettiğini test eder."""
    gercekler = [
        {"box": [10.0, 10.0, 50.0, 50.0], "class_id": 0},
        {"box": [100.0, 100.0, 150.0, 150.0], "class_id": 0},
    ]
    tahminler = [
        {"box": [10.0, 10.0, 50.0, 50.0], "score": 0.95, "class_id": 0},
        {"box": [100.0, 100.0, 150.0, 150.0], "score": 0.90, "class_id": 0},
    ]

    ap, prec, rec = MAPHesaplayici.sinif_ap_hesapla(tahminler, gercekler, iou_esigi=0.5)
    assert ap == pytest.approx(1.0)
    assert prec[-1] == pytest.approx(1.0)
    assert rec[-1] == pytest.approx(1.0)


def test_kapsamli_map_coco():
    """Çok sınıflı mAP@0.5 ve COCO mAP@0.5:0.95 hesaplamasını test eder."""
    gercekler = [
        {"box": [10.0, 10.0, 50.0, 50.0], "class_id": 0},
        {"box": [100.0, 100.0, 150.0, 150.0], "class_id": 1},
    ]
    tahminler = [
        {"box": [10.0, 10.0, 50.0, 50.0], "score": 0.95, "class_id": 0},
        {"box": [100.0, 100.0, 150.0, 150.0], "score": 0.90, "class_id": 1},
    ]
    siniflar = ["A", "B"]

    sonuc = MAPHesaplayici.kapsamli_map_hesapla(tahminler, gercekler, siniflar)
    assert sonuc["map_05"] == pytest.approx(1.0)
    assert sonuc["map_05_95"] == pytest.approx(1.0)
    assert "A" in sonuc["sinif_ap_05"]


def test_yolo_yoneticisi_cikarim():
    """YOLOYoneticisi sınıfının başlatılmasını ve çıkarım çıktı şemasını test eder."""
    yolo = YOLOYoneticisi(model_adi="yolov8n.pt")
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)

    tespitler = yolo.cikarim_yap(dummy_img, conf=0.1, iou=0.45)
    assert isinstance(tespitler, list)


def test_dashboard_gorsellestirici(tmp_path):
    """4 panelli YOLO teşhis panosunun oluşturulduğunu test eder."""
    kayip_gecmisi = {"box_loss": [1.0, 0.8], "cls_loss": [1.2, 0.9], "dfl_loss": [1.1, 0.8]}
    map_bilgisi = {
        "map_05": 0.85,
        "map_05_95": 0.65,
        "sinif_ap_05": {"Vida": 0.90, "Somun": 0.80},
        "sinif_ap_coco": {"Vida": 0.70, "Somun": 0.60},
        "pr_egrileri": {
            "Vida": (np.array([1.0, 0.9]), np.array([0.5, 1.0])),
            "Somun": (np.array([1.0, 0.8]), np.array([0.5, 1.0])),
        },
    }
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    tahminler = [{"box": [10.0, 10.0, 50.0, 50.0], "score": 0.92, "class_id": 0, "class_name": "Vida"}]
    siniflar = ["Vida", "Somun"]

    hedef = tmp_path / "test_yolo_paneli.png"
    cikti = YOLOGorsellestirici.dashboard_ciz(
        kayip_gecmisi, map_bilgisi, dummy_img, tahminler, siniflar, hedef_dosya=hedef
    )

    assert cikti.exists()
    assert cikti.stat().st_size > 0
