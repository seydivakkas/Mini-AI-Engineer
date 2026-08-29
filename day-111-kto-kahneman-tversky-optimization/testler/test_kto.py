"""
Kahneman-Tversky Optimization (KTO) Testleri (Day 111).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.kto_kaybi import KTOLoss, hesapla_dizi_logprob
from src.kto_modeli import KTODilModeli
from src.kto_laboratuvari import KTOLaboratuvari
from src.gorsellestirici import KTOGorsellestirici


def test_kto_loss_temel_hesabi():
    """Politika ve referans eşitken KTO kaybının pozitif ve sonlu olduğunu test eder."""
    loss_fn = KTOLoss(beta=0.1, lambda_d=1.0, lambda_u=1.33)

    pi_logps = torch.tensor([-5.0, -10.0, -8.0, -12.0])
    ref_logps = torch.tensor([-5.0, -10.0, -8.0, -12.0])
    labels = torch.tensor([1.0, 1.0, -1.0, -1.0])

    loss, metrikler = loss_fn(pi_logps, ref_logps, labels)

    assert loss.item() > 0.0
    assert "toplam_kayip" in metrikler
    assert "kayip_d" in metrikler
    assert "kayip_u" in metrikler


def test_kto_loss_asimetri_kayip_kacinmasi():
    """lambda_u > lambda_d nedeniyle kayıp kaçınmasının asimetrik cezalandırma yaptığını test eder."""
    loss_fn = KTOLoss(beta=0.5, lambda_d=1.0, lambda_u=2.0)

    # Durum 1: Yalnızca Desirable örneğinde hata (r = -2, z_ref = 0)
    pi_d = torch.tensor([-4.0])
    ref_d = torch.tensor([0.0])
    lbl_d = torch.tensor([1.0])
    loss_d, _ = loss_fn(pi_d, ref_d, lbl_d)

    # Durum 2: Yalnızca Undesirable örneğinde eşit büyüklükte hata (r = +2, z_ref = 0)
    pi_u = torch.tensor([0.0])
    ref_u = torch.tensor([-4.0])
    lbl_u = torch.tensor([-1.0])
    loss_u, _ = loss_fn(pi_u, ref_u, lbl_u)

    # lambda_u (2.0) > lambda_d (1.0) olduğu için loss_u > loss_d olmalıdır!
    assert loss_u.item() > loss_d.item()


def test_kto_loss_z_ref_hareketli_ortalama():
    """Eğitim sırasında z_ref referans noktasının hareketli ortalamayla güncellendiğini test eder."""
    loss_fn = KTOLoss(beta=0.1)
    loss_fn.train()

    pi_logps = torch.tensor([10.0, 10.0])
    ref_logps = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 1.0])

    # Başlangıçta z_ref = 0
    assert float(loss_fn.z_ref.item()) == 0.0

    loss_fn(pi_logps, ref_logps, labels)
    # Güncelleme sonrası z_ref > 0 olmalı
    assert float(loss_fn.z_ref.item()) > 0.0


def test_hesapla_dizi_logprob():
    """hesapla_dizi_logprob fonksiyonunun maskeli log-olasılık hesabını test eder."""
    logits = torch.randn(2, 6, 20)
    labels = torch.randint(0, 19, (2, 6))
    maske = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    logps = hesapla_dizi_logprob(logits, labels, maske)

    assert logps.shape == (2,)
    assert not torch.isnan(logps).any()


def test_kto_dil_modeli_forward():
    """KTODilModeli forward geçişini ve çıktı şeklini test eder."""
    model = KTODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (3, 16))

    logits = model(x)
    assert logits.shape == (3, 16, 100)


def test_kto_dil_modeli_logprob():
    """KTODilModeli logprob_hesapla fonksiyonunu test eder."""
    model = KTODilModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    x = torch.randint(0, 99, (2, 12))
    mask = torch.ones_like(x, dtype=torch.float32)

    logps = model.logprob_hesapla(x, mask)
    assert logps.shape == (2,)


def test_kto_laboratuvari_egitim():
    """KTOLaboratuvari tekil ikili veri üretimi ve KTO eğitim döngüsünü test eder."""
    lab = KTOLaboratuvari(vocab_size=200, dim=64, num_heads=2, num_layers=2, cihaz=torch.device("cpu"))
    input_ids, maske, etiketler = lab.tekil_ikili_veri_uret(ornek_sayisi=32, prompt_len=4, resp_len=6)

    assert input_ids.shape == (32, 10)
    assert etiketler.shape == (32,)

    rapor = lab.kto_egit(input_ids, maske, etiketler, epok_sayisi=3, batch_size=16, lr=1e-3)

    assert len(rapor["toplam_kayiplar"]) == 3
    assert rapor["toplam_kayiplar"][-1] < rapor["toplam_kayiplar"][0]  # Kayıp düşmeli
    assert rapor["marjinler"][-1] > 0.0                                # Marjin pozitifleşmeli


def test_kto_gorsellestirici_pano():
    """KTOGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = KTOGorsellestirici(dpi=100)
    ornek_egitim = {
        "toplam_kayiplar": [0.85, 0.55, 0.35, 0.18],
        "kayiplar_d": [0.4, 0.25, 0.15, 0.08],
        "kayiplar_u": [0.45, 0.30, 0.20, 0.10],
        "dogruluklar": [50.0, 75.0, 90.0, 99.0],
        "r_d_ort": [0.0, 0.4, 1.1, 1.7],
        "r_u_ort": [0.0, -0.3, -0.9, -1.5],
        "marjinler": [0.0, 0.7, 2.0, 3.2],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_kto_pano.png")
        gorsellestirici.pano_olustur(ornek_egitim, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
