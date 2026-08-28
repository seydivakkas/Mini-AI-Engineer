"""
Day 69: Optimizer ve Scheduler Laboratuvarı Birim Test Paketi
============================================================
Lion optimizer algoritması, Linear Warmup Cosine zamanlayıcısı,
parametre gruplama, model mimarisi ve görselleştirici testleri.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from src.lion_optimizer import Lion
from src.zamanlayicilar import LinearWarmupCosineScheduler
from src.laboratuvar_modeli import DeneySinirAgi, parametre_gruplari_ayristir
from src.optimizer_laboratuvari import OptimizerLaboratuvari
from src.gorsellestirici import OptimizerLaboratuvarGorsellestirici


@pytest.fixture
def gecici_dizin():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_lion_optimizer_adim_ve_parametre_guncelleme():
    """Lion optimizer'ın gradyan adımı attığını ve durum tensörünü oluşturduğunu test eder."""
    p = nn.Parameter(torch.tensor([1.0, -1.0], requires_grad=True))
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)

    loss = (p ** 2).sum()
    loss.backward()
    opt.step()

    # Parametrelerin sıfıra doğru güncellendiğini doğrula
    assert p[0].item() < 1.0
    assert p[1].item() > -1.0
    assert "exp_avg" in opt.state[p]


def test_lion_optimizer_gecersiz_parametreler():
    """Lion'un geçersiz hiperparametrelerde ValueError fırlattığını test eder."""
    p = nn.Parameter(torch.randn(2))
    with pytest.raises(ValueError):
        Lion([p], lr=-0.01)
    with pytest.raises(ValueError):
        Lion([p], betas=(1.2, 0.99))
    with pytest.raises(ValueError):
        Lion([p], weight_decay=-0.1)


def test_lion_decoupled_weight_decay():
    """Lion'da ayrıştırılmış weight decay'in ağırlığı küçülttüğünü test eder."""
    p1 = nn.Parameter(torch.tensor([10.0]))
    p1.grad = torch.tensor([0.0])  # Sıfır gradyan
    opt = Lion([p1], lr=0.1, weight_decay=0.1)
    opt.step()

    # p_new = p * (1 - lr * wd) - lr * sign(0) = 10 * (1 - 0.01) = 9.9
    assert p1.item() < 10.0


def test_linear_warmup_cosine_scheduler_egrisi():
    """Scheduler'ın warmup ve cosine fazlarında doğru LR değerleri ürettiğini test eder."""
    p = nn.Parameter(torch.randn(2))
    opt = torch.optim.SGD([p], lr=0.1)
    sch = LinearWarmupCosineScheduler(opt, warmup_epochs=2, max_epochs=10, eta_min=0.001)

    # 1. Warmup adımları (Doğrusal artış)
    lr_0 = sch.get_lr()[0]
    sch.step()
    lr_1 = sch.get_lr()[0]
    assert lr_1 > lr_0

    # 2. Warmup bitişi sonrası (Sönümleme)
    sch.step()
    sch.step()
    lr_after = sch.get_lr()[0]
    assert lr_after <= 0.1


def test_parametre_gruplari_ayristirma():
    """Parametre ayırıcı fonksiyonun bias ve norm katmanlarını weight decay'den muaf tuttuğunu test eder."""
    model = DeneySinirAgi(girdi_kanali=3, sinif_sayisi=2, taban_kanal=8)
    gruplar = parametre_gruplari_ayristir(model, weight_decay=0.05)

    assert len(gruplar) == 2
    assert gruplar[0]["weight_decay"] == 0.05  # Ağırlıklar
    assert gruplar[1]["weight_decay"] == 0.0   # Bias & BatchNorm

    # Tüm parametrelerin eksiksiz dağıtıldığını doğrula
    toplam_param_grup = sum(len(g["params"]) for g in gruplar)
    assert toplam_param_grup == len(list(model.parameters()))


def test_deney_sinir_agi_ileri_yayilim():
    """Deney sinir ağının tensör boyutlarını doğru ürettiğini test eder."""
    model = DeneySinirAgi(girdi_kanali=3, sinif_sayisi=5, taban_kanal=16)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 5)


def test_optimizer_laboratuvari_tek_deney():
    """Tekil deney koşusunun sorunsuz çalıştığını ve metrik ürettiğini test eder."""
    sonuc = OptimizerLaboratuvari.tek_deney_kos(
        deney_adi="TestDeney",
        optimizer_turu="lion",
        scheduler_turu="warmup_cosine",
        lr=1e-4,
        weight_decay=0.01,
        toplam_epoch=2,
        warmup_epoch=1
    )

    assert sonuc["son_train_loss"] > 0
    assert len(sonuc["gecmis"]["train_loss"]) == 2
    assert sonuc["tahmini_opt_bellek_kb"] > 0


def test_gorsellestirici_pano_kaydi(gecici_dizin: str):
    """Görselleştiricinin teşhis panosu görselini ürettiğini test eder."""
    panel_yolu = os.path.join(gecici_dizin, "test_opt_paneli.png")
    sonuclar = OptimizerLaboratuvari.tum_laboratuvari_kos(toplam_epoch=2)

    cizim_yolu = OptimizerLaboratuvarGorsellestirici.panoyu_ciz_ve_kaydet(
        laboratuvar_sonuclari=sonuclar,
        cikti_yolu=panel_yolu
    )

    assert os.path.exists(cizim_yolu)
    assert os.path.getsize(cizim_yolu) > 10000
