"""
Odds Ratio Preference Optimization (ORPO) Testleri (Day 112).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.orpo_kaybi import ORPOLoss, sayisal_kararli_log_odds, hesapla_token_bazli_logprob
from src.orpo_modeli import ORPODilModeli
from src.orpo_laboratuvari import ORPOLaboratuvari
from src.gorsellestirici import ORPOGorsellestirici


def test_sayisal_kararli_log_odds():
    """sayisal_kararli_log_odds fonksiyonunun sayısal taşma yapmadan çalıştığını test eder."""
    logps = torch.tensor([-0.1, -0.6931, -2.0, -10.0])
    log_odds = sayisal_kararli_log_odds(logps)

    # p = 0.5 (logp = -0.6931) iken odds = 1.0 -> log(odds) ~ 0.0
    assert abs(float(log_odds[1].item())) < 0.01
    assert not torch.isnan(log_odds).any()


def test_orpo_loss_temel_hesap():
    """ORPOLoss fonksiyonunun SFT ve OR kayıplarını birleştirdiğini test eder."""
    loss_fn = ORPOLoss(lambda_or=0.5)

    c_logps = torch.tensor([[-0.5, -0.5, -0.5]])
    r_logps = torch.tensor([[-2.0, -2.0, -2.0]])
    c_mask = torch.tensor([[0.0, 1.0, 1.0]])
    r_mask = torch.tensor([[0.0, 1.0, 1.0]])

    loss, metrikler = loss_fn(c_logps, r_logps, c_mask, r_mask)

    assert loss.item() > 0.0
    assert "toplam_kayip" in metrikler
    assert "kayip_sft" in metrikler
    assert "kayip_or" in metrikler
    assert metrikler["dogruluk"].item() == 1.0


def test_orpo_loss_odds_ayrisma():
    """Chosen log-olasılığı yüksek olduğunda log-odds oranının pozitifleştiğini test eder."""
    loss_fn = ORPOLoss(lambda_or=1.0)

    c_logps = torch.tensor([[-0.2, -0.2]])
    r_logps = torch.tensor([[-5.0, -5.0]])
    mask = torch.tensor([[1.0, 1.0]])

    _, metrikler = loss_fn(c_logps, r_logps, mask, mask)
    assert metrikler["log_odds_ratio"].item() > 0.0


def test_hesapla_token_bazli_logprob():
    """hesapla_token_bazli_logprob fonksiyonunun token adımı log-olasılıklarını doğru hesapladığını test eder."""
    logits = torch.randn(2, 6, 20)
    labels = torch.randint(0, 19, (2, 6))

    per_token_logps = hesapla_token_bazli_logprob(logits, labels)

    assert per_token_logps.shape == (2, 6)
    assert not torch.isnan(per_token_logps).any()


def test_orpo_dil_modeli_forward():
    """ORPODilModeli forward geçişini ve çıktı şeklini test eder."""
    model = ORPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (3, 16))

    logits = model(x)
    assert logits.shape == (3, 16, 100)


def test_orpo_dil_modeli_token_logprob():
    """ORPODilModeli token_logprob_hesapla fonksiyonunu test eder."""
    model = ORPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (2, 12))

    logps = model.token_logprob_hesapla(x)
    assert logps.shape == (2, 12)


def test_orpo_laboratuvari_egitim():
    """ORPOLaboratuvari tek aşamalı monolitik eğitim döngüsünü test eder."""
    lab = ORPOLaboratuvari(vocab_size=200, dim=64, num_heads=2, num_layers=2, cihaz=torch.device("cpu"))
    c_ids, r_ids, c_mask, r_mask = lab.sentetik_tercih_verisi_uret(cift_sayisi=32, prompt_len=4, resp_len=6)

    assert c_ids.shape == (32, 10)
    assert r_ids.shape == (32, 10)

    rapor = lab.orpo_egit(c_ids, r_ids, c_mask, r_mask, epok_sayisi=3, batch_size=16, lr=1e-3)

    assert len(rapor["toplam_kayiplar"]) == 3
    assert rapor["toplam_kayiplar"][-1] < rapor["toplam_kayiplar"][0]  # Kayıp düşmeli
    assert rapor["dogruluklar"][-1] >= rapor["dogruluklar"][0]        # Doğruluk artmalı


def test_orpo_gorsellestirici_pano():
    """ORPOGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = ORPOGorsellestirici(dpi=100)
    ornek_egitim = {
        "toplam_kayiplar": [1.5, 0.9, 0.5, 0.2],
        "kayiplar_sft": [1.2, 0.7, 0.4, 0.15],
        "kayiplar_or": [0.6, 0.4, 0.2, 0.1],
        "log_odds_oranlari": [0.0, 1.5, 3.2, 4.8],
        "dogruluklar": [50.0, 75.0, 90.0, 99.0],
    }
    ornek_kiyas = {
        "ppo_gpu_model": 4,
        "dpo_gpu_model": 2,
        "orpo_gpu_model": 1,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_orpo_pano.png")
        gorsellestirici.pano_olustur(ornek_egitim, ornek_kiyas, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
