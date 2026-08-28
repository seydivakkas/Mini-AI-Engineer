"""
Day 30: Büyük Final Çoklu Görev Görsel Platformu Birim Testleri.
"""

import os
import numpy as np
import pytest
import torch

from src.coklu_gorev_modeli import CokluGorevGorselModeli, BelirsizlikAgirlikliKayip
from src.model_optimizasyoncusu import ModelOptimizasyoncusu
from src.takip_ve_analitik_motoru import CokluGorevTakipAnalitikMotoru
from src.platform_yoneticisi import PlatformYoneticisi
from src.gorsellestirici import BuyukFinalGorsellestirici


def test_coklu_gorev_modeli_ileri_besleme():
    """Çoklu görev modeli 4 başlık ileri besleme ve tensör boyutları testi."""
    model = CokluGorevGorselModeli(
        in_channels=3,
        num_scene_classes=4,
        num_object_classes=4,
        num_seg_classes=5,
        reid_dim=128
    )
    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert out["scene_logits"].shape == (2, 4)
    assert out["det_cls"].shape == (2, 4, 16, 16)
    assert out["det_box"].shape == (2, 4, 16, 16)
    assert out["det_obj"].shape == (2, 1, 16, 16)
    assert out["seg_logits"].shape == (2, 5, 128, 128)
    assert out["reid_embedding"].shape == (2, 128)


def test_belirsizlik_agirlikli_kayip():
    """Homoscedastic belirsizlik ağırlıklı kayıp ve geri yayılım testi."""
    kayip_mod = BelirsizlikAgirlikliKayip(gorev_sayisi=3)
    l1 = torch.tensor(1.5, requires_grad=True)
    l2 = torch.tensor(2.0, requires_grad=True)
    l3 = torch.tensor(0.8, requires_grad=True)

    toplam_loss, agirliklar = kayip_mod([l1, l2, l3])
    assert len(agirliklar) == 3
    assert toplam_loss.item() > 0

    toplam_loss.backward()
    assert kayip_mod.log_vars.grad is not None


def test_model_optimizasyoncusu_kuantizasyon():
    """Model kuantizasyonu ve performans kıyaslama testi."""
    model = CokluGorevGorselModeli()
    sonuclar = ModelOptimizasyoncusu.performans_kıyasla(
        model=model,
        girdi_sekli=(1, 3, 64, 64),
        tekrar_sayisi=3,
        cihaz="cpu"
    )

    assert "FP32" in sonuclar
    assert "FP16" in sonuclar
    assert "INT8" in sonuclar
    assert sonuclar["FP32"]["fps"] > 0
    assert sonuclar["INT8"]["boyut_mb"] < sonuclar["FP32"]["boyut_mb"]


def test_takip_ve_analitik_motoru():
    """Entegre takip ve mekansal hız kestirim testi."""
    motor = CokluGorevTakipAnalitikMotoru(max_cosine_dist=0.5, max_age=5)

    kutular1 = [np.array([10.0, 10.0, 40.0, 80.0])]
    emb1 = np.random.randn(1, 128).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    t1 = motor.guncelle(kutular1, emb1, [1])
    assert len(t1) == 1
    assert t1[0].track_id == 1

    kutular2 = [np.array([16.0, 10.0, 46.0, 80.0])]
    t2 = motor.guncelle(kutular2, emb1, [1])
    assert len(t2) == 1
    assert t2[0].track_id == 1
    assert t2[0].hiz_gecmisi[-1] == pytest.approx(6.0, 0.1)


def test_platform_yoneticisi_entegrasyon():
    """Uçtan uca platform telemetri üretim testi."""
    platform = PlatformYoneticisi(device="cpu")
    kare = np.full((128, 128, 3), 100, dtype=np.uint8)
    nesneler = [{"box": [20.0, 30.0, 60.0, 80.0], "sinif_id": 1}]

    telemetri = platform.isle_kare(kare, nesneler)
    assert "sahne_etiketi" in telemetri
    assert telemetri["seg_maskesi"].shape == (128, 128)
    assert len(telemetri["aktif_takipciler"]) == 1


def test_buyuk_final_gorsellestirici(tmp_path):
    """6 panelli büyük final teşhis panosu çizim testi."""
    kare = np.full((128, 128, 3), 120, dtype=np.uint8)
    platform = PlatformYoneticisi(device="cpu")
    telemetri = platform.isle_kare(kare, [{"box": [10.0, 20.0, 40.0, 60.0], "sinif_id": 1}])

    opt_res = {
        "FP32": {"fps": 50.0, "boyut_mb": 18.0},
        "FP16": {"fps": 90.0, "boyut_mb": 9.0},
        "INT8": {"fps": 130.0, "boyut_mb": 4.5}
    }
    belirsizlik = {"cls_sigma": [1.0, 0.9], "det_sigma": [1.2, 1.1], "seg_sigma": [1.5, 1.3]}
    radar = {"siniflandirma_acc": 0.95, "tespit_map": 0.93, "bolutleme_miou": 0.88, "takip_mota": 0.96, "takip_idf1": 0.97}

    out_file = str(tmp_path / "test_final.png")
    cizim_path = BuyukFinalGorsellestirici.buyuk_final_panosu_ciz(
        ornek_kare_rgb=kare,
        telemetri=telemetri,
        optimizasyon_sonuclari=opt_res,
        belirsizlik_gecmisi=belirsizlik,
        radar_metrikleri=radar,
        hedef_path=out_file
    )
    assert os.path.exists(cizim_path)
