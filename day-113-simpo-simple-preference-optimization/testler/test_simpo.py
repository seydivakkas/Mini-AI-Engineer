"""
Simple Preference Optimization (SimPO) Testleri (Day 113).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.simpo_kaybi import SimPOLoss, hesapla_token_bazli_logprob
from src.simpo_modeli import SimPODilModeli
from src.simpo_laboratuvari import SimPOLaboratuvari
from src.gorsellestirici import SimPOGorsellestirici


def test_simpo_loss_temel_hesap():
    """SimPOLoss fonksiyonunun temel marjin ve kayıp hesaplamasını test eder."""
    loss_fn = SimPOLoss(beta=2.0, gamma=0.5)

    c_logps = torch.tensor([[-0.5, -0.5, -0.5]])
    r_logps = torch.tensor([[-2.0, -2.0, -2.0]])
    c_mask = torch.tensor([[0.0, 1.0, 1.0]])
    r_mask = torch.tensor([[0.0, 1.0, 1.0]])

    loss, metrikler = loss_fn(c_logps, r_logps, c_mask, r_mask)

    assert loss.item() > 0.0
    assert "kayip" in metrikler
    assert "odul_farki" in metrikler
    assert "marjin_ihlali" in metrikler
    assert metrikler["dogruluk"].item() == 1.0


def test_simpo_hedef_marjin_etkisi():
    """Hedef marjin (gamma > 0) olduğunda kaybın gamma=0 durumuna göre daha yüksek çıktığını test eder."""
    loss_fn_no_margin = SimPOLoss(beta=2.0, gamma=0.0)
    loss_fn_margin = SimPOLoss(beta=2.0, gamma=1.0)

    c_logps = torch.tensor([[-0.5, -0.5]])
    r_logps = torch.tensor([[-1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    loss_0, _ = loss_fn_no_margin(c_logps, r_logps, mask, mask)
    loss_m, _ = loss_fn_margin(c_logps, r_logps, mask, mask)

    assert loss_m.item() > loss_0.item()


def test_simpo_uzunluk_normalizasyonu():
    """Uzunluk normalizasyonunun farklı uzunluktaki yanıtlarda ortalama token logprobunu doğru aldığını test eder."""
    loss_fn = SimPOLoss(beta=1.0, gamma=0.0)

    # 2 tokenli vs 4 tokenli dizi, aynı token başına logprob (-1.0)
    c_logps = torch.tensor([[-1.0, -1.0, 0.0, 0.0]])
    r_logps = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    c_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    r_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    _, metrikler = loss_fn(c_logps, r_logps, c_mask, r_mask)

    # Her ikisinin ortalama logp'si -1.0 olduğu için ödül farkı 0 olmalı
    assert abs(float(metrikler["odul_farki"].item())) < 1e-5


def test_hesapla_token_bazli_logprob():
    """hesapla_token_bazli_logprob fonksiyonunun token adımı log-olasılıklarını doğru hesapladığını test eder."""
    logits = torch.randn(2, 6, 20)
    labels = torch.randint(0, 19, (2, 6))

    per_token_logps = hesapla_token_bazli_logprob(logits, labels)

    assert per_token_logps.shape == (2, 6)
    assert not torch.isnan(per_token_logps).any()


def test_simpo_dil_modeli_forward():
    """SimPODilModeli forward geçişini ve çıktı şeklini test eder."""
    model = SimPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (3, 16))

    logits = model(x)
    assert logits.shape == (3, 16, 100)


def test_simpo_dil_modeli_token_logprob():
    """SimPODilModeli token_logprob_hesapla fonksiyonunu test eder."""
    model = SimPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (2, 12))

    logps = model.token_logprob_hesapla(x)
    assert logps.shape == (2, 12)


def test_simpo_laboratuvari_egitim():
    """SimPOLaboratuvari referans modelsiz eğitim döngüsünü test eder."""
    lab = SimPOLaboratuvari(vocab_size=200, dim=64, num_heads=2, num_layers=2, cihaz=torch.device("cpu"))
    c_ids, r_ids, c_mask, r_mask = lab.sentetik_tercih_verisi_uret(cift_sayisi=32, prompt_len=4, resp_len=6)

    assert c_ids.shape == (32, 10)
    assert r_ids.shape == (32, 10)

    rapor = lab.simpo_egit(c_ids, r_ids, c_mask, r_mask, epok_sayisi=3, batch_size=16, lr=1e-3)

    assert len(rapor["kayiplar"]) == 3
    assert rapor["kayiplar"][-1] < rapor["kayiplar"][0]  # Kayıp düşmeli
    assert rapor["dogruluklar"][-1] >= rapor["dogruluklar"][0]  # Doğruluk artmalı


def test_simpo_gorsellestirici_pano():
    """SimPOGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = SimPOGorsellestirici(dpi=100)
    ornek_egitim = {
        "kayiplar": [1.5, 0.9, 0.5, 0.2],
        "chosen_odulleri": [-2.0, -1.2, -0.8, -0.4],
        "rejected_odulleri": [-2.0, -2.5, -3.5, -4.8],
        "odul_farklari": [0.0, 1.3, 2.7, 4.4],
        "marjin_ihlalleri": [100.0, 20.0, 0.0, 0.0],
        "dogruluklar": [50.0, 85.0, 98.0, 100.0],
    }
    ornek_kiyas = {
        "yontemler": ["PPO", "DPO", "ORPO", "SimPO"],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_simpo_pano.png")
        gorsellestirici.pano_olustur(ornek_egitim, ornek_kiyas, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
