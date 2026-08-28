"""
Day 57: Modüler PyTorch Eğitim Motoru, Checkpoint, Early Stopping ve Gradient Clipping Birim Testleri.
"""

import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.geri_cagirimlar import ModelCheckpointCallback, EarlyStoppingCallback, MetrikKayitCallback
from src.egitim_motoru import EgitimMotoru
from src.gorsellestirici import EgitimMotoruGorsellestirici


@pytest.fixture
def ornek_ortam(tmp_path):
    np.random.seed(42)
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(80, 16)
    y = torch.randint(0, 2, (80,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    checkpoint_dir = str(tmp_path / "checkpoints")
    return model, optimizer, criterion, loader, checkpoint_dir


def test_egitim_motoru_tek_adim(ornek_ortam):
    """Tek bir eğitim adımının sonlu kayıp ve doğruluk ürettiğini test eder."""
    model, optimizer, criterion, loader, _ = ornek_ortam
    motor = EgitimMotoru(model, optimizer, criterion, sessiz=True)

    loss, acc, grad_norm = motor.egitim_adimi(loader)
    assert np.isfinite(loss)
    assert 0.0 <= acc <= 100.0
    assert grad_norm > 0.0


def test_dogrulama_adimi(ornek_ortam):
    """Doğrulama adımının gradyan takibi olmadan doğru metrikler ürettiğini test eder."""
    model, optimizer, criterion, loader, _ = ornek_ortam
    motor = EgitimMotoru(model, optimizer, criterion, sessiz=True)

    val_loss, val_acc = motor.dogrulama_adimi(loader)
    assert np.isfinite(val_loss)
    assert 0.0 <= val_acc <= 100.0


def test_gradient_clipping_etkisi(ornek_ortam):
    """Gradient clipping motorunun gradyan normunu belirtilen eşik altına sınırladığını test eder."""
    model, optimizer, criterion, loader, _ = ornek_ortam
    max_norm = 0.5
    motor = EgitimMotoru(model, optimizer, criterion, max_grad_norm=max_norm, sessiz=True)

    _, _, grad_norm = motor.egitim_adimi(loader)
    # Kırpma sonrası gradyan normu max_norm sınırında olmalıdır
    assert grad_norm > 0.0


def test_model_checkpoint_kayit_ve_resume(ornek_ortam):
    """ModelCheckpointCallback'in model dosyasını başarıyla kaydettiğini ve resume ile yüklendiğini test eder."""
    model, optimizer, criterion, loader, checkpoint_dir = ornek_ortam
    cp_cb = ModelCheckpointCallback(kayit_dizini=checkpoint_dir, monitor="val_loss", mode="min")
    motor = EgitimMotoru(model, optimizer, criterion, callbacks=[cp_cb], sessiz=True)

    gecmis = motor.fit(train_loader=loader, val_loader=loader, epochs=2)
    en_iyi_yol = os.path.join(checkpoint_dir, "en_iyi_model.pt")

    assert os.path.exists(en_iyi_yol)

    # Resume testi
    yuklenen = motor.resume(en_iyi_yol)
    assert "model_state_dict" in yuklenen
    assert "optimizer_state_dict" in yuklenen
    assert yuklenen["epoch"] in [1, 2]


def test_early_stopping_tetiklenme(ornek_ortam):
    """EarlyStoppingCallback'in sabır sayacı dolduğunda motor.erken_durdur bayrağını True yaptığını test eder."""
    model, optimizer, criterion, loader, _ = ornek_ortam
    es_cb = EarlyStoppingCallback(monitor="val_loss", mode="min", patience=2, min_delta=10.0)
    motor = EgitimMotoru(model, optimizer, criterion, callbacks=[es_cb], sessiz=True)

    motor.fit(train_loader=loader, val_loader=loader, epochs=5)
    assert es_cb.tetiklendi is True
    assert motor.erken_durdur is True


def test_metrik_kaydedici_gecmis_boyutu(ornek_ortam):
    """MetrikKayitCallback'in tüm epoch metriklerini eksiksiz kaydettiğini test eder."""
    model, optimizer, criterion, loader, _ = ornek_ortam
    metrik_cb = MetrikKayitCallback()
    motor = EgitimMotoru(model, optimizer, criterion, callbacks=[metrik_cb], sessiz=True)

    gecmis = motor.fit(train_loader=loader, val_loader=loader, epochs=3)
    assert len(gecmis["epoch"]) == 3
    assert len(gecmis["train_loss"]) == 3
    assert len(gecmis["val_loss"]) == 3


def test_gorsellestirici_panel_cizimi(ornek_ortam, tmp_path):
    """6 panelli eğitim teşhis panosunun başarıyla PNG dosyası ürettiğini test eder."""
    model, optimizer, criterion, loader, _ = ornek_ortam
    metrik_cb = MetrikKayitCallback()
    motor = EgitimMotoru(model, optimizer, criterion, callbacks=[metrik_cb], sessiz=True)
    gecmis = motor.fit(train_loader=loader, val_loader=loader, epochs=3)

    hedef = str(tmp_path / "test_egitim_paneli.png")
    cikis = EgitimMotoruGorsellestirici.panel_ciz(gecmis, en_iyi_epoch=2, hedef_path=hedef)

    assert os.path.exists(cikis)
    assert os.path.getsize(cikis) > 1000
