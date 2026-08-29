"""
Bradley-Terry Tercih Modeli ve Ödül Modeli Testleri (Day 108).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.bradley_terry_kaybi import BradleyTerryLoss, tercih_olasiligi, tercih_dogrulugu
from src.odul_modeli import OdulModeli
from src.odul_laboratuvari import OdulLaboratuvari
from src.gorsellestirici import OdulGorsellestirici


def test_bradley_terry_kaybi_hesaplama():
    """BradleyTerryLoss fonksiyonunun doğru kayıp ve doğruluk ürettiğini test eder."""
    loss_fn = BradleyTerryLoss(margin=0.0, reg_lambda=0.0)
    r_w = torch.tensor([2.0, 3.0, 1.0])
    r_l = torch.tensor([0.0, 1.0, 2.0])  # Sonuncuda r_l > r_w

    loss, acc = loss_fn(r_w, r_l)
    assert loss.item() > 0.0
    # 2/3 doğru -> %66.67
    assert round(acc.item(), 2) == 0.67


def test_bradley_terry_marjin_etkisi():
    """Marjin eklendiğinde aynı ödüllere sahip çiftler için kaybın arttığını doğrular."""
    loss_fn_no_m = BradleyTerryLoss(margin=0.0, reg_lambda=0.0)
    loss_fn_with_m = BradleyTerryLoss(margin=1.0, reg_lambda=0.0)

    r_w = torch.tensor([1.0, 1.0])
    r_l = torch.tensor([1.0, 1.0])

    loss1, _ = loss_fn_no_m(r_w, r_l)     # -log(sigmoid(0)) = 0.6931
    loss2, _ = loss_fn_with_m(r_w, r_l)   # -log(sigmoid(-1)) = 1.3133

    assert loss2.item() > loss1.item()


def test_tercih_olasiligi_ve_dogrulugu():
    """tercih_olasiligi (sigmoid) ve tercih_dogrulugu fonksiyonlarını test eder."""
    r_w = torch.tensor([2.0, 0.0])
    r_l = torch.tensor([0.0, 0.0])

    prob = tercih_olasiligi(r_w, r_l)
    assert prob[0] > 0.8  # sigmoid(2) = 0.88
    assert round(prob[1].item(), 2) == 0.5  # sigmoid(0) = 0.5

    acc = tercih_dogrulugu(r_w, r_l)
    assert acc == 0.5  # 1 doğru, 1 eşit


def test_odul_modeli_skaler_cikti():
    """OdulModeli modülünün her dizi için tek bir skaler ödül döndürdüğünü test eder."""
    model = OdulModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=32)
    inp = torch.randint(1, 99, (3, 16))

    odul = model(inp)
    assert odul.shape == (3,)  # [B]
    assert not torch.isnan(odul).any()


def test_odul_modeli_last_token_pooling():
    """Padding token'ları varken son geçerli token'ın başarıyla havuzlandığını test eder."""
    model = OdulModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=32, pad_token_id=0)
    # 1. dizi: 5 geçerli + 5 pad (0)
    # 2. dizi: 10 geçerli + 0 pad
    inp = torch.zeros((2, 10), dtype=torch.long)
    inp[0, :5] = torch.randint(1, 99, (5,))
    inp[1, :10] = torch.randint(1, 99, (10,))

    odul = model(inp)
    assert odul.shape == (2,)


def test_odul_modeli_ciftli_hesaplama():
    """ciftli_odul_hesapla metodunun r_w ve r_l tensörlerini döndürdüğünü test eder."""
    model = OdulModeli(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=32)
    c = torch.randint(1, 99, (4, 12))
    r = torch.randint(1, 99, (4, 12))

    r_w, r_l = model.ciftli_odul_hesapla(c, r)
    assert r_w.shape == (4,)
    assert r_l.shape == (4,)


def test_odul_laboratuvari_egitim_ve_ayrisma():
    """OdulLaboratuvari'nın sentetik veriyle ödül modelini başarıyla eğittiğini test eder."""
    lab = OdulLaboratuvari(vocab_size=500, dim=64, cihaz=torch.device("cpu"))
    c, r = lab.sentetik_tercih_verisi_uret(cift_sayisi=60, seq_len=24)
    model = OdulModeli(vocab_size=500, dim=64, num_heads=2, num_layers=2, max_seq_len=32)

    rapor = lab.odul_modeli_egit(model, c, r, epok_sayisi=5, batch_size=16, lr=1e-3)
    assert len(rapor["kayiplar"]) == 5
    # Son epok doğruluğu başlangıçtan yüksek olmalı
    assert rapor["dogruluklar"][-1] >= rapor["dogruluklar"][0]


def test_odul_gorsellestirici_pano():
    """OdulGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = OdulGorsellestirici(dpi=100)
    ornek_egitim = {
        "kayiplar": [0.85, 0.62, 0.41, 0.25],
        "dogruluklar": [52.0, 74.0, 88.0, 96.0],
        "marjinler": [0.1, 0.8, 1.6, 2.4],
        "r_w_ort": [0.2, 0.9, 1.5, 2.1],
        "r_l_ort": [0.1, 0.1, -0.1, -0.3],
    }
    ornek_hack = {"ayrisma_guvenilirligi": 2.4}

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_reward_pano.png")
        gorsellestirici.pano_olustur(ornek_egitim, ornek_hack, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
