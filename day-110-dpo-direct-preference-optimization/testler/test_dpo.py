"""
Direct Preference Optimization (DPO) Testleri (Day 110).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import math
import pytest
import torch

from src.dpo_kaybi import DPOLoss, hesapla_dizi_logprob
from src.dpo_modeli import DPODilModeli
from src.dpo_laboratuvari import DPOLaboratuvari
from src.gorsellestirici import DPOGorsellestirici


def test_dpo_loss_temel_hesabi():
    """Politika ve referans eşitken DPO kaybının log(2) ~ 0.6931 olduğunu test eder."""
    loss_fn = DPOLoss(beta=0.1)

    pi_c = torch.tensor([-5.0, -10.0])
    pi_r = torch.tensor([-5.0, -10.0])
    ref_c = torch.tensor([-5.0, -10.0])
    ref_r = torch.tensor([-5.0, -10.0])

    loss, metrikler = loss_fn(pi_c, pi_r, ref_c, ref_r)

    assert abs(float(loss.item()) - math.log(2.0)) < 1e-4
    assert metrikler["dogruluk"].item() == 0.0  # Logit = 0 -> Dogruluk = 0
    assert metrikler["ortuk_marjin"].item() == 0.0


def test_dpo_loss_ayrisma_ve_marjin():
    """Chosen log-olasılığı arttıkça örtük marjinin büyüdüğünü ve kaybın düştüğünü test eder."""
    loss_fn = DPOLoss(beta=0.5)

    pi_c = torch.tensor([0.0])
    pi_r = torch.tensor([-10.0])
    ref_c = torch.tensor([-5.0])
    ref_r = torch.tensor([-5.0])

    loss, metrikler = loss_fn(pi_c, pi_r, ref_c, ref_r)

    # pi_logratio = 10, ref_logratio = 0 -> logits = 0.5 * 10 = 5.0
    # loss = -log(sigmoid(5.0)) ~ 0.0067
    assert loss.item() < 0.05
    assert metrikler["dogruluk"].item() == 1.0
    assert metrikler["ortuk_marjin"].item() == 5.0


def test_dpo_loss_uzunluk_norm():
    """Uzunluk normalizasyonu seçeneğinin çalıştığını test eder."""
    loss_fn = DPOLoss(beta=0.1, uzunluk_norm=True)

    pi_c = torch.tensor([-10.0])
    pi_r = torch.tensor([-20.0])
    ref_c = torch.tensor([-10.0])
    ref_r = torch.tensor([-20.0])

    c_lens = torch.tensor([10])
    r_lens = torch.tensor([20])

    loss, metrikler = loss_fn(pi_c, pi_r, ref_c, ref_r, c_lens, r_lens)
    assert loss.item() > 0.0


def test_hesapla_dizi_logprob():
    """hesapla_dizi_logprob fonksiyonunun kaydırma ve maskelemeyi doğru uyguladığını test eder."""
    logits = torch.randn(2, 6, 20)
    labels = torch.randint(0, 19, (2, 6))
    maske = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    logps = hesapla_dizi_logprob(logits, labels, maske)

    assert logps.shape == (2,)
    assert not torch.isnan(logps).any()


def test_dpo_dil_modeli_forward():
    """DPODilModeli forward geçişini ve çıktı şeklini test eder."""
    model = DPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (3, 16))

    logits = model(x)
    assert logits.shape == (3, 16, 100)


def test_dpo_dil_modeli_logprob():
    """DPODilModeli logprob_hesapla fonksiyonunu test eder."""
    model = DPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (2, 12))
    mask = torch.ones_like(x, dtype=torch.float32)

    logps = model.logprob_hesapla(x, mask)
    assert logps.shape == (2,)


def test_dpo_laboratuvari_egitim():
    """DPOLaboratuvari sentetik tercih verisi üretimi ve DPO eğitim döngüsünü test eder."""
    lab = DPOLaboratuvari(vocab_size=200, dim=64, num_heads=2, num_layers=2, cihaz=torch.device("cpu"))
    c_ids, r_ids, c_mask, r_mask = lab.sentetik_tercih_verisi_uret(cift_sayisi=32, prompt_len=4, resp_len=6)

    assert c_ids.shape == (32, 10)
    assert r_ids.shape == (32, 10)

    rapor = lab.dpo_egit(c_ids, r_ids, c_mask, r_mask, epok_sayisi=3, batch_size=16, lr=1e-3)

    assert len(rapor["kayiplar"]) == 3
    assert rapor["kayiplar"][-1] < rapor["kayiplar"][0]  # Kayıp düşmeli
    assert rapor["marjinler"][-1] > 0.0                  # Marjin pozitifleşmeli


def test_dpo_gorsellestirici_pano():
    """DPOGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = DPOGorsellestirici(dpi=100)
    ornek_egitim = {
        "kayiplar": [0.69, 0.45, 0.25, 0.12],
        "dogruluklar": [50.0, 75.0, 90.0, 99.0],
        "r_w_ort": [0.0, 0.5, 1.2, 1.8],
        "r_l_ort": [0.0, -0.4, -1.0, -1.6],
        "marjinler": [0.0, 0.9, 2.2, 3.4],
    }
    ornek_kiyas = {
        "ppo_model_sayisi": 4,
        "dpo_model_sayisi": 2,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_dpo_pano.png")
        gorsellestirici.pano_olustur(ornek_egitim, ornek_kiyas, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
