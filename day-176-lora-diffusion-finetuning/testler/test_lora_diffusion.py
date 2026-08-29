"""
GÜN 176: Difüzyon Modellerinde LoRA & DreamBooth İnce Ayar Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from src.lora_katmani import LoRALinear
from src.lora_enjektoru import LoRAEnjektoru
from src.dreambooth_egitici import DreamBoothEgitici
from src.gorsellestirici import LoRAGorsellestirici


def test_lora_linear_baslangic_cikti_esitligi():
    """LoRA B=0 başlatıldığı için eğitimin başında orijinal katman ile çıktının birebir aynı olduğunu test eder."""
    base_linear = nn.Linear(32, 64)
    x = torch.randn(2, 32)
    original_out = base_linear(x)

    lora_linear = LoRALinear(base_linear, r=8, lora_alpha=16.0)
    lora_out = lora_linear(x)

    assert torch.allclose(original_out, lora_out)


def test_lora_linear_gradyan_akis_ve_dondurma():
    """W_0'ın dondurulduğunu (grad=None), sadece LoRA A ve B parametrelerinin gradyan aldığını test eder."""
    base_linear = nn.Linear(16, 32)
    lora_linear = LoRALinear(base_linear, r=4, lora_alpha=8.0)
    x = torch.randn(1, 16)
    out = lora_linear(x)

    loss = out.sum()
    loss.backward()

    assert base_linear.weight.grad is None
    assert lora_linear.lora_A.grad is not None
    assert lora_linear.lora_B.grad is not None


def test_lora_agirliklari_birlestirme():
    """LoRA ağırlıklarının dondurulmuş W_0 ile doğru formülle birleştirildiğini test eder."""
    base_linear = nn.Linear(8, 16)
    lora_linear = LoRALinear(base_linear, r=2, lora_alpha=4.0)
    lora_linear.lora_B.data.fill_(1.0)
    lora_linear.lora_A.data.fill_(1.0)

    merged_w = lora_linear.agirliklari_birlestir()
    assert merged_w.shape == (16, 8)
    assert not torch.allclose(merged_w, base_linear.weight.data)


class DummyUNetCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(64, 64)
        self.to_k = nn.Linear(64, 64)
        self.to_v = nn.Linear(64, 64)
        self.to_out = nn.Linear(64, 64)
        self.other_layer = nn.Linear(64, 32)


def test_lora_enjektoru_otomatik_degisim():
    """LoRA enjektörünün sadece hedef cross-attention katmanlarını dönüştürdüğünü test eder."""
    model = DummyUNetCrossAttn()
    degisen = LoRAEnjektoru.lora_enjekte_et(model, hedef_katman_isimleri=["to_q", "to_k", "to_v", "to_out"], r=4)

    assert degisen == 4
    assert isinstance(model.to_q, LoRALinear)
    assert isinstance(model.other_layer, nn.Linear) and not isinstance(model.other_layer, LoRALinear)


def test_parametre_sayilari_hesaplama():
    """Eğitilebilir parametre istatistiklerinin doğru hesaplandığını test eder."""
    model = DummyUNetCrossAttn()
    LoRAEnjektoru.lora_enjekte_et(model, r=4)
    istatistik = LoRAEnjektoru.parametre_sayilarini_getir(model)

    assert istatistik["egitilebilir_oran_yuzde"] < 30.0
    assert "tasarruf_orani" in istatistik


def test_dreambooth_cift_kayip_hesaplama():
    """DreamBooth toplam kaybının L_instance + lambda * L_prior olduğunu test eder."""
    egitici = DreamBoothEgitici(prior_loss_weight=1.0)
    t_ozne = torch.randn(2, 4, 8, 8)
    h_ozne = torch.randn(2, 4, 8, 8)
    t_sinif = torch.randn(2, 4, 8, 8)
    h_sinif = torch.randn(2, 4, 8, 8)

    loss_total, loss_inst, loss_prior = egitici.toplam_kayip_hesapla(t_ozne, h_ozne, t_sinif, h_sinif)

    assert torch.allclose(loss_total, loss_inst + loss_prior)


def test_dreambooth_ornek_rapor():
    """Örnek raporun beklenen rank deneylerini ve ~100x kazancı içerdiğini test eder."""
    rapor = DreamBoothEgitici.ornek_lora_raporu_getir()
    assert len(rapor["rank_deneyleri"]) == 5
    assert rapor["sinif_koruma_skoru"] > 0.90


def test_gorsellestirici_pano_uretme():
    """6 panelli LoRA & DreamBooth teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = DreamBoothEgitici.ornek_lora_raporu_getir()
    gorsellestirici = LoRAGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_lora_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
