"""
Day 71: Çökmeye Dayanıklı Checkpoint ve Eğitim Motoru Birim Test Paketi
======================================================================
Atomik I/O, durum restorasyonu, RNG determinizmi, Top-K budama ve görselleştirici testleri.
"""

import os
import tempfile
import random
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.model import KompaktVisionNet
from src.checkpoint_yoneticisi import GuvenliCheckpointYoneticisi
from src.egitim_motoru import DevamEdebilirEgitimMotoru
from src.gorsellestirici import CheckpointTeshisGorsellestirici


@pytest.fixture
def gecici_dizin():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_atomik_kayit_ve_dosya_varligi(gecici_dizin: str):
    """Atomik kaydın hedef dosyayı ürettiğini ve geçici .tmp dosyası bırakmadığını test eder."""
    yonetici = GuvenliCheckpointYoneticisi(kayit_dizini=gecici_dizin, maks_saklanan=3)
    model = KompaktVisionNet(girdi_kanali=3, sinif_sayisi=2, taban_kanal=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2)

    kayit_yolu = yonetici.kaydet_atomik(
        epoch=1, model=model, optimizer=opt, scheduler=sched, val_loss=0.5, val_acc=80.0
    )

    assert os.path.exists(kayit_yolu)
    assert not os.path.exists(f"{kayit_yolu}.tmp")
    assert os.path.exists(os.path.join(gecici_dizin, "last.pt"))
    assert os.path.exists(os.path.join(gecici_dizin, "best.pt"))


def test_tam_durum_geri_yukleme_model_opt_sched(gecici_dizin: str):
    """Model, Optimizer ve Scheduler durumlarının eksiksiz geri yüklendiğini test eder."""
    yonetici = GuvenliCheckpointYoneticisi(kayit_dizini=gecici_dizin)
    model = KompaktVisionNet(girdi_kanali=3, sinif_sayisi=2, taban_kanal=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    # 1 adım ilerlet
    x = torch.randn(2, 3, 32, 32)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    sched.step()

    kayit_yolu = yonetici.kaydet_atomik(
        epoch=1, model=model, optimizer=opt, scheduler=sched, val_loss=0.3, val_acc=90.0
    )

    # Yeni boş nesneler oluştur ve geri yükle
    yeni_model = KompaktVisionNet(girdi_kanali=3, sinif_sayisi=2, taban_kanal=8)
    yeni_opt = torch.optim.AdamW(yeni_model.parameters(), lr=1e-3)
    yeni_sched = torch.optim.lr_scheduler.StepLR(yeni_opt, step_size=1, gamma=0.5)

    durum = yonetici.yukle_ve_geri_yukle(
        dosya_yolu=kayit_yolu,
        model=yeni_model,
        optimizer=yeni_opt,
        scheduler=yeni_sched,
        cihaz=torch.device("cpu")
    )

    assert durum["epoch"] == 1
    assert durum["val_loss"] == 0.3
    # Scheduler LR'ının korunduğunu test et (gamma=0.5 uygulandı -> 0.0005)
    assert yeni_opt.param_groups[0]["lr"] == 0.0005


def test_rng_durum_toplama_ve_geri_yukleme():
    """RNG durum restorasyonunun tam deterministik rastgele sayı devamlılığı sağladığını test eder."""
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # Durumu kaydet
    rng_state = GuvenliCheckpointYoneticisi.rng_durumu_topla()

    # Birkaç rastgele sayı üret
    r1 = random.random()
    n1 = float(np.random.rand())
    t1 = float(torch.rand(1).item())

    # Rastgele sayıları başka tohumla boz
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    # Durumu geri yükle
    GuvenliCheckpointYoneticisi.rng_durumu_geri_yukle(rng_state)

    r2 = random.random()
    n2 = float(np.random.rand())
    t2 = float(torch.rand(1).item())

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


def test_top_k_checkpoint_budama(gecici_dizin: str):
    """Maksimum saklanan sınırını aşan eski checkpointlerin diskten temizlendiğini test eder."""
    yonetici = GuvenliCheckpointYoneticisi(kayit_dizini=gecici_dizin, maks_saklanan=2)
    model = KompaktVisionNet(girdi_kanali=3, sinif_sayisi=2, taban_kanal=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    # 4 epoch kaydet (farklı loss değerleriyle)
    p1 = yonetici.kaydet_atomik(1, model, opt, sched, val_loss=0.8, val_acc=50.0)
    p2 = yonetici.kaydet_atomik(2, model, opt, sched, val_loss=0.4, val_acc=70.0)
    p3 = yonetici.kaydet_atomik(3, model, opt, sched, val_loss=0.6, val_acc=60.0)
    p4 = yonetici.kaydet_atomik(4, model, opt, sched, val_loss=0.2, val_acc=90.0)

    # Top-2 en iyi loss: epoch 4 (0.2) ve epoch 2 (0.4) saklanmalı, epoch 1 (0.8) silinmeli
    assert os.path.exists(p4)
    assert os.path.exists(p2)
    assert not os.path.exists(p1)


def test_best_ve_last_pt_dosyalari(gecici_dizin: str):
    """best.pt'nin en düşük loss değerini sakladığını test eder."""
    yonetici = GuvenliCheckpointYoneticisi(kayit_dizini=gecici_dizin)
    model = KompaktVisionNet(girdi_kanali=3, sinif_sayisi=2, taban_kanal=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    yonetici.kaydet_atomik(1, model, opt, sched, val_loss=0.5, val_acc=70.0)
    yonetici.kaydet_atomik(2, model, opt, sched, val_loss=0.8, val_acc=60.0)

    best_paket = torch.load(os.path.join(gecici_dizin, "best.pt"), weights_only=False)
    assert best_paket["epoch"] == 1
    assert best_paket["val_loss"] == 0.5


def test_olmayan_checkpoint_dosyasi_hata_firlatma(gecici_dizin: str):
    """Olmayan bir dosya istendiğinde FileNotFoundError fırlatıldığını test eder."""
    yonetici = GuvenliCheckpointYoneticisi(kayit_dizini=gecici_dizin)
    model = KompaktVisionNet(girdi_kanali=3, sinif_sayisi=2, taban_kanal=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    with pytest.raises(FileNotFoundError):
        yonetici.yukle_ve_geri_yukle(
            "olmayan_dosya.pt", model, opt, sched, torch.device("cpu")
        )


def test_cokus_ve_devam_etme_dongusu(gecici_dizin: str):
    """Çöküş simülasyonu ve ardından başarılı devam etme döngüsünü test eder."""
    motor = DevamEdebilirEgitimMotoru(kayit_dizini=gecici_dizin, lr=1e-3)

    # 1. Aşama: Epoch 1'de simüle çökme
    with pytest.raises(RuntimeError):
        motor.egit(hedef_epoch=3, cokus_epochu=1)

    # 2. Aşama: Geri yükle ve devam et
    motor_2 = DevamEdebilirEgitimMotoru(kayit_dizini=gecici_dizin, lr=1e-3)
    yeni_ep = motor_2.checkpointten_devam_et(os.path.join(gecici_dizin, "last.pt"))
    assert yeni_ep == 2

    sonuc = motor_2.egit(hedef_epoch=3)
    assert sonuc["son_epoch"] == 3


def test_gorsellestirici_pano_uretimi(gecici_dizin: str):
    """Görselleştiricinin teşhis panosu görselini ürettiğini test eder."""
    panel_yolu = os.path.join(gecici_dizin, "test_resumable_paneli.png")
    gecmis = {
        "epoch": [1, 2, 3, 4],
        "train_loss": [1.5, 1.2, 0.9, 0.7],
        "val_loss": [1.4, 1.1, 0.8, 0.6],
        "val_accuracy": [40.0, 55.0, 70.0, 85.0],
        "lr": [0.001, 0.0008, 0.0005, 0.0002]
    }

    cizim_yolu = CheckpointTeshisGorsellestirici.panoyu_ciz_ve_kaydet(
        egitim_gecmisi=gecmis,
        cokus_epochu=2,
        cikti_yolu=panel_yolu
    )

    assert os.path.exists(cizim_yolu)
    assert os.path.getsize(cizim_yolu) > 10000
