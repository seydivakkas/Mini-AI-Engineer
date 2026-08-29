"""
PyTest Birim Testleri - Day 183: DeepSpeed ZeRO-1/2/3 ve CPU/NVMe Bellek Boşaltma.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.zero_offload_motoru import OffloadDevice, ZeROOffloadYapilandirma, CPUAdamWOptimizer
from src.zero_infinity_yonetici import ZeROInfinityKatmanSarmalayici, ZeROOffloadProfilleyici
from src.gorsellestirici import ZeROGorsellestirici


@pytest.fixture
def basit_linear_model():
    """Test için 64x128 doğrusal model."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )


def test_zero_offload_yapilandirma():
    """1. ZeROOffloadYapilandirma doğru parametreleri ve sözlük çıktısını üretmelidir."""
    config = ZeROOffloadYapilandirma(
        stage=3,
        offload_optimizer_device=OffloadDevice.CPU,
        offload_param_device=OffloadDevice.NVME,
        pin_memory=True,
        buffer_count=2,
    )
    d = config.to_dict()
    assert d["zero_stage"] == 3
    assert d["offload_optimizer"] == "CPU"
    assert d["offload_param"] == "NVME"
    assert d["double_buffering"] is True


def test_cpu_adamw_optimizer_guncelleme(basit_linear_model):
    """2. CPUAdamWOptimizer parametreleri doğru matematiksel kuralla güncellemelidir."""
    optimizer = CPUAdamWOptimizer(
        params=list(basit_linear_model.parameters()),
        lr=0.01,
        weight_decay=0.01,
    )

    x = torch.randn(4, 64)
    target = torch.randn(4, 32)
    out_once = basit_linear_model(x)
    loss = nn.functional.mse_loss(out_once, target)
    loss.backward()

    # Ağırlıkların güncellenmeden önceki kopyası
    ilk_agirlik = basit_linear_model[0].weight.data.clone()

    optimizer.step()

    guncel_agirlik = basit_linear_model[0].weight.data
    # Ağırlıklar değişmiş olmalıdır
    assert not torch.allclose(ilk_agirlik, guncel_agirlik)


def test_cpu_adamw_master_weights_ve_momentler(basit_linear_model):
    """3. FP32 master ağırlıklar, momentum ve variance tamponları CPU üzerinde saklanmalıdır."""
    optimizer = CPUAdamWOptimizer(
        params=list(basit_linear_model.parameters()),
        lr=0.01,
    )

    for mw in optimizer.master_weights:
        assert mw.device.type == "cpu"
        assert mw.dtype == torch.float32

    for m in optimizer.exp_avg:
        assert m.device.type == "cpu"
        assert m.dtype == torch.float32

    for v in optimizer.exp_avg_sq:
        assert v.device.type == "cpu"
        assert v.dtype == torch.float32


def test_cpu_adamw_zero_grad(basit_linear_model):
    """4. zero_grad() çağrısı gradyanları sıfırlamalıdır."""
    optimizer = CPUAdamWOptimizer(params=list(basit_linear_model.parameters()))
    x = torch.randn(4, 64)
    out = basit_linear_model(x)
    loss = out.sum()
    loss.backward()

    assert basit_linear_model[0].weight.grad is not None
    optimizer.zero_grad()
    assert torch.allclose(basit_linear_model[0].weight.grad, torch.zeros_like(basit_linear_model[0].weight.grad))


def test_zero_infinity_katman_sarmalama():
    """5. ZeROInfinityKatmanSarmalayici ağırlıkları CPU'da tutmalı ve ileri geçişte çalıştırmalıdır."""
    lin = nn.Linear(32, 64)
    inf_layer = ZeROInfinityKatmanSarmalayici(
        module=lin,
        offload_device=OffloadDevice.CPU,
        compute_device="cpu",
    )

    x = torch.randn(8, 32)
    out = inf_layer(x)
    assert out.shape == (8, 64)
    # İleri geçiş bittiğinde ağırlıklar temizlenmiş olmalıdır
    assert getattr(inf_layer.module, "weight", None) is None


def test_zero_offload_profilleyici_matematiksel_dogruluk():
    """6. ZeROOffloadProfilleyici statik bellek formülleri (16B, 4B, 12B) tutarlı olmalıdır."""
    res = ZeROOffloadProfilleyici.bellek_dagilimi_hesapla_gb(param_milyar=70.0)

    # 70B Model: 70 * 16 = 1120 GB (yaklaşık 1043.1 GiB)
    assert res["ddp_gpu_vram_gb"] == pytest.approx(1043.08, rel=0.05)
    # ZeRO-Offload GPU yükü (Param 2B + Grad 2B = 4B -> toplamın %25'i)
    assert res["zero_offload_gpu_gb"] == pytest.approx(res["ddp_gpu_vram_gb"] * 0.25, rel=0.05)
    # ZeRO-Offload CPU yükü (AdamW 12B -> toplamın %75'i)
    assert res["zero_offload_cpu_gb"] == pytest.approx(res["ddp_gpu_vram_gb"] * 0.75, rel=0.05)
    assert res["offload_vram_tasarrufu_yuzde"] == pytest.approx(75.0, abs=1.0)


def test_zero_offload_coklu_model_raporu():
    """7. Çoklu model raporu 7B'den 1T'ye kadar 5 modeli eksiksiz içermelidir."""
    rapor = ZeROOffloadProfilleyici.coklu_model_profil_raporu()
    assert len(rapor) == 5
    adlar = [m["model_adi"] for m in rapor]
    assert "Llama-2-7B" in adlar
    assert "Llama-3-70B" in adlar
    assert "Titan-1T (1000B)" in adlar


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. ZeROGorsellestirici 6 panelli teşhis panosunu başarıyla oluşturmalıdır."""
    cikti = str(tmp_path / "test_zero_paneli.png")
    rapor = ZeROOffloadProfilleyici.coklu_model_profil_raporu()

    ZeROGorsellestirici.zero_offload_paneli_olustur(
        profil_raporu=rapor,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
