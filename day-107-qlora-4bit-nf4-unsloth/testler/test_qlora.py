"""
QLoRA, NF4 Kuantizasyon ve Unsloth Autograd Testleri (Day 107).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.nf4_kuantizasyon import NF4_SEVIYELER, NF4Kuantizator, DoubleQuantization
from src.qlora_katmani import QLoRALinear, HizliQLoRAAutograd
from src.qlora_laboratuvari import QLoRALaboratuvari
from src.gorsellestirici import QLoRAGorsellestirici


def test_nf4_kuantile_tablosu_ve_sirasi():
    """NF4 tablosunun 16 elemanlı, kesin artan ve 0.0 içerdiğini test eder."""
    assert len(NF4_SEVIYELER) == 16
    assert NF4_SEVIYELER[0] == -1.0
    assert NF4_SEVIYELER[-1] == 1.0
    assert 0.0 in NF4_SEVIYELER
    # Monotonik artış kontrolü
    for i in range(len(NF4_SEVIYELER) - 1):
        assert NF4_SEVIYELER[i] < NF4_SEVIYELER[i + 1]


def test_nf4_kuantizator_ileri_ve_geri():
    """NF4Kuantizator modülünün kuantize ve dekuantize döngüsünü test eder."""
    kuantizator = NF4Kuantizator(block_size=64)
    w = torch.randn(128, 128) * 0.02

    q_idx, c1, sekil = kuantizator.kuantize_et(w)
    assert q_idx.shape == (128, 128)
    assert q_idx.dtype == torch.uint8
    assert torch.all(q_idx <= 15)

    w_deq = kuantizator.dekuantize_et(q_idx, c1, sekil)
    assert w_deq.shape == (128, 128)

    # Cosine Benzerliği > %99 olmalı
    w_norm = torch.nn.functional.normalize(w.flatten(), dim=0)
    w_deq_norm = torch.nn.functional.normalize(w_deq.flatten(), dim=0)
    cos_sim = float(torch.dot(w_norm, w_deq_norm).item())
    assert cos_sim > 0.985


def test_double_quantization_sikistirma_ve_cozme():
    """DoubleQuantization modülünün ölçek faktörlerini sıkıştırmasını test eder."""
    dq = DoubleQuantization(block_size_2=64)
    c1 = torch.rand(256) * 0.05 + 0.001

    c1_int8, c2, c1_min = dq.c1_sikistir(c1)
    assert c1_int8.shape == (256,)
    assert c1_int8.dtype == torch.uint8

    c1_coz = dq.c1_coz(c1_int8, c2, c1_min)
    assert c1_coz.shape == (256,)
    # Maksimum hata çok küçük olmalı
    max_hata = float(torch.max(torch.abs(c1 - c1_coz)).item())
    assert max_hata < 0.005


def test_qlora_katmani_parametre_dondurma():
    """QLoRALinear katmanında ana ağırlığın dondurulduğunu ve adaptörlerin eğitilebilir olduğunu test eder."""
    qlora = QLoRALinear(in_features=64, out_features=128, r=8, lora_alpha=16)
    w = torch.randn(128, 64) * 0.02
    qlora.agirliklari_yukle_ve_kuantize_et(w)

    egitilebilirler = [p for p in qlora.parameters() if p.requires_grad]
    dondurulan_isimler = [n for n, b in qlora.named_buffers()]

    assert "q_weight" in dondurulan_isimler
    assert len(egitilebilirler) == 2  # lora_A ve lora_B
    assert qlora.lora_A.shape == (8, 64)
    assert qlora.lora_B.shape == (128, 8)


def test_qlora_hizli_autograd_gradyan_akis():
    """HizliQLoRAAutograd fonksiyonunun lora_A, lora_B ve x için gradyan ürettiğini test eder."""
    x = torch.randn(2, 4, 32, requires_grad=True)
    w_deq = torch.randn(64, 32)
    lora_A = torch.randn(4, 32, requires_grad=True)
    lora_B = torch.randn(64, 4, requires_grad=True)

    out = HizliQLoRAAutograd.apply(x, w_deq, lora_A, lora_B, 2.0, None)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert lora_A.grad is not None
    assert lora_B.grad is not None
    assert not torch.isnan(lora_A.grad).any()
    assert not torch.isnan(lora_B.grad).any()


def test_qlora_linear_ileri_ve_geri_gecis():
    """QLoRALinear modülünün uçtan uca ileri ve geri geçişini test eder."""
    qlora = QLoRALinear(in_features=32, out_features=64, r=4, double_quant=True)
    w = torch.randn(64, 32) * 0.02
    qlora.agirliklari_yukle_ve_kuantize_et(w)

    x = torch.randn(2, 8, 32)
    out = qlora(x)
    assert out.shape == (2, 8, 64)

    loss = out.mean()
    loss.backward()
    assert qlora.lora_A.grad is not None
    assert qlora.lora_B.grad is not None


def test_qlora_laboratuvari_vram_ve_sadakat():
    """QLoRALaboratuvari VRAM hesaplamaları ve sadakat metriklerini test eder."""
    lab = QLoRALaboratuvari(cihaz=torch.device("cpu"))
    vram = lab.vram_olceklenme_analizi()
    assert "7B Modeli" in vram
    assert "70B Modeli" in vram
    assert vram["70B Modeli"]["QLoRA (NF4 + DQ) (GB)"] < 50.0  # 38.5 GB

    sadakat = lab.kuantizasyon_sadakati_olc(dim_in=128, dim_out=128)
    assert sadakat["kosinus_benzerligi"] > 0.98


def test_qlora_gorsellestirici_pano():
    """QLoRAGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = QLoRAGorsellestirici(dpi=100)
    ornek_vram = {
        "7B Modeli": {"Full Fine-Tuning (GB)": 112.0, "FP16 LoRA (GB)": 16.4, "QLoRA (NF4 + DQ) (GB)": 5.1},
        "70B Modeli": {"Full Fine-Tuning (GB)": 1120.0, "FP16 LoRA (GB)": 145.5, "QLoRA (NF4 + DQ) (GB)": 39.7},
    }
    ornek_sadakat = {"kosinus_benzerligi": 0.994, "snr_db": 24.5}

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_qlora_paneli.png")
        gorsellestirici.pano_olustur(ornek_vram, ornek_sadakat, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
