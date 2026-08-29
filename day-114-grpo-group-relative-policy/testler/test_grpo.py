"""
Group Relative Policy Optimization (GRPO) Testleri (Day 114).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.grpo_kaybi import GRPOLoss, grup_goreli_avantaj_hesapla, hesapla_token_bazli_logprob
from src.grpo_modeli import GRPODilModeli
from src.grpo_laboratuvari import GRPOLaboratuvari
from src.gorsellestirici import GRPOGorsellestirici


def test_grup_goreli_avantaj_hesapla():
    """grup_goreli_avantaj_hesapla fonksiyonunun sıfır ortalama ve birim varyans ürettiğini test eder."""
    oduller = torch.tensor([1.0, 0.5, 1.5, 0.0, 2.0])
    avantajlar = grup_goreli_avantaj_hesapla(oduller)

    assert abs(float(avantajlar.mean().item())) < 1e-5
    assert abs(float(avantajlar.std(unbiased=False).item()) - 1.0) < 1e-3


def test_grpo_loss_temel_hesap():
    """GRPOLoss fonksiyonunun temel taşıyıcı kayıp ve metrik hesaplamasını test eder."""
    loss_fn = GRPOLoss(clip_eps=0.2, beta_kl=0.04)

    G, S = 4, 10
    logp_theta = torch.randn(G, S)
    logp_old = logp_theta.clone()
    logp_ref = logp_theta.clone()
    oduller = torch.tensor([1.0, 0.0, 1.5, 0.5])
    token_mask = torch.ones(G, S)

    loss, metrikler = loss_fn(logp_theta, logp_old, logp_ref, oduller, token_mask)

    assert not torch.isnan(loss)
    assert "toplam_kayip" in metrikler
    assert "politika_kaybi" in metrikler
    assert "kl_kaybi" in metrikler
    assert "ortalama_odul" in metrikler


def test_grpo_kl_schulman_estimator():
    """Politika referans modele eşit olduğunda KL sapmasının sıfır çıktığını test eder."""
    loss_fn = GRPOLoss(clip_eps=0.2, beta_kl=1.0)

    G, S = 2, 5
    logp = torch.randn(G, S)
    oduller = torch.tensor([1.0, 1.0])
    token_mask = torch.ones(G, S)

    _, metrikler = loss_fn(logp, logp, logp, oduller, token_mask)

    assert abs(float(metrikler["kl_kaybi"].item())) < 1e-5


def test_hesapla_token_bazli_logprob():
    """hesapla_token_bazli_logprob fonksiyonunun token adımı log-olasılıklarını doğru hesapladığını test eder."""
    logits = torch.randn(2, 6, 20)
    labels = torch.randint(0, 19, (2, 6))

    per_token_logps = hesapla_token_bazli_logprob(logits, labels)

    assert per_token_logps.shape == (2, 6)
    assert not torch.isnan(per_token_logps).any()


def test_grpo_dil_modeli_forward():
    """GRPODilModeli forward geçişini ve çıktı şeklini test eder."""
    model = GRPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (3, 16))

    logits = model(x)
    assert logits.shape == (3, 16, 100)


def test_grpo_dil_modeli_grup_ornekle():
    """GRPODilModeli grup_ornekle fonksiyonunun G adet paralel yanıt ürettiğini test eder."""
    model = GRPODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    prompt = torch.randint(0, 99, (1, 6))

    grup_ciktilar = model.grup_ornekle(prompt, group_size=4, max_new_tokens=8)

    assert grup_ciktilar.shape == (4, 14)  # 6 + 8 = 14


def test_grpo_laboratuvari_egitim():
    """GRPOLaboratuvari Critic'siz grup örneklemeli akıl yürütme eğitim döngüsünü test eder."""
    lab = GRPOLaboratuvari(vocab_size=100, dim=64, num_heads=2, num_layers=2, cihaz=torch.device("cpu"))
    rapor = lab.grpo_egit(prompt_sayisi=8, group_size=4, epok_sayisi=3, lr=1e-3)

    assert len(rapor["toplam_kayiplar"]) == 3
    assert len(rapor["ortalama_oduller"]) == 3
    assert not any(torch.isnan(torch.tensor(rapor["ortalama_oduller"])))


def test_grpo_gorsellestirici_pano():
    """GRPOGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = GRPOGorsellestirici(dpi=100)
    ornek_egitim = {
        "toplam_kayiplar": [0.5, 0.2, -0.1, -0.4],
        "politika_kayiplari": [0.4, 0.1, -0.2, -0.5],
        "kl_kayiplari": [0.01, 0.05, 0.1, 0.15],
        "ortalama_oduller": [0.2, 0.6, 1.1, 1.4],
        "std_oduller": [0.1, 0.2, 0.3, 0.1],
        "kirpilma_oranlari": [2.0, 5.0, 8.0, 12.0],
    }
    ornek_kiyas = {
        "kriterler": ["Critic Modeli (V)", "VRAM Kullanımı"],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_grpo_pano.png")
        gorsellestirici.pano_olustur(ornek_egitim, ornek_kiyas, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
