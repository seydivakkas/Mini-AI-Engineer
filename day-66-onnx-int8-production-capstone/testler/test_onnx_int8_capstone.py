"""
Day 66: ONNX & INT8 Capstone Birim Test Paketi
==============================================
PyTorch mimarisi, ONNX ihracatı, dinamik batch desteği, INT8 kuantizasyon,
sayısal eşdeğerlik ve benchmark motorunun doğrulanması.
"""

import os
import tempfile
import pytest
import numpy as np
import torch

from src.model_mimari import UretimVisionNet
from src.onnx_aktarici import ONNXDonusturucu
from src.kuantizasyon_motoru import INT8Kuantizator
from src.cikarim_motoru import ONNXInferenceEngine
from src.karsilastirici_benchmark import ModelBenchmarkKarsilastirici
from src.gorsellestirici import CapstoneGorsellestirici


@pytest.fixture
def ornek_pytorch_model() -> UretimVisionNet:
    model = UretimVisionNet(girdi_kanali=3, sinif_sayisi=5, taban_kanal=16)
    model.eval()
    return model


@pytest.fixture
def gecici_dizin():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_model_mimarisi_ileri_yayilim(ornek_pytorch_model: UretimVisionNet):
    """PyTorch modelinin girdi ve cikti tensör boyutlarını test eder."""
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = ornek_pytorch_model(x)
    assert out.shape == (2, 5)
    assert not torch.isnan(out).any()


def test_onnx_export_ve_dogrulama(ornek_pytorch_model: UretimVisionNet, gecici_dizin: str):
    """PyTorch modelinin ONNX'e başarıyla aktarılıp doğrulandığını test eder."""
    onnx_yolu = os.path.join(gecici_dizin, "test_model.onnx")
    ornek_tensör = torch.randn(1, 3, 32, 32)

    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(
        model=ornek_pytorch_model,
        ornek_girdi=ornek_tensör,
        cikti_yolu=onnx_yolu
    )

    assert os.path.exists(onnx_yolu)
    ozet = aktarici.model_ozeti_al(onnx_yolu)
    assert ozet["dugum_sayisi"] > 0
    assert ozet["opset_versiyonu"] == 18
    assert "girdi_gorsel" in ozet["girdi_isimleri"]


def test_dinamik_eksen_batch_destegi(ornek_pytorch_model: UretimVisionNet, gecici_dizin: str):
    """ONNX modelinin dinamik batch boyutlarında (B=1, B=3, B=6) çalıştığını test eder."""
    onnx_yolu = os.path.join(gecici_dizin, "dinamik_test.onnx")
    ornek_tensör = torch.randn(1, 3, 32, 32)

    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(
        model=ornek_pytorch_model,
        ornek_girdi=ornek_tensör,
        cikti_yolu=onnx_yolu
    )

    motor = ONNXInferenceEngine(onnx_yolu)
    for b in [1, 3, 6]:
        girdi = np.random.randn(b, 3, 32, 32).astype(np.float32)
        cikti = motor.tahmin_et(girdi)
        assert cikti.shape == (b, 5)


def test_int8_dinamik_kuantizasyon(ornek_pytorch_model: UretimVisionNet, gecici_dizin: str):
    """ONNX modelinin INT8'e başarıyla kuantize edildiğini ve sıkıştırıldığını test eder."""
    onnx_fp32 = os.path.join(gecici_dizin, "model_fp32.onnx")
    onnx_int8 = os.path.join(gecici_dizin, "model_int8.onnx")

    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(ornek_pytorch_model, torch.randn(1, 3, 32, 32), onnx_fp32)

    kuantizator = INT8Kuantizator()
    sonuc = kuantizator.dinamik_kuantize_et(onnx_fp32, onnx_int8)

    assert os.path.exists(onnx_int8)
    assert sonuc["sikistirma_orani"] > 1.0
    assert sonuc["int8_boyut_mb"] <= sonuc["fp32_boyut_mb"]


def test_onnx_inference_engine_ve_isinma(ornek_pytorch_model: UretimVisionNet, gecici_dizin: str):
    """Inference engine'in ısınma ve gecikme ölçümlerini test eder."""
    onnx_yolu = os.path.join(gecici_dizin, "engine_test.onnx")
    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(ornek_pytorch_model, torch.randn(1, 3, 32, 32), onnx_yolu)

    motor = ONNXInferenceEngine(onnx_yolu, is_parcacigi_sayisi=2)
    motor.isinma_yap(ornek_sekil=(1, 3, 32, 32), tekrar=3)

    girdi = np.random.randn(2, 3, 32, 32).astype(np.float32)
    profil = motor.gecikme_olcumle(girdi, tekrar_sayisi=15)

    assert "ortalama_ms" in profil
    assert "p95_ms" in profil
    assert profil["ortalama_ms"] > 0


def test_sayisal_esdegerlik_kosinus_benzerligi(ornek_pytorch_model: UretimVisionNet, gecici_dizin: str):
    """PyTorch, ONNX FP32 ve ONNX INT8 arasındaki sayısal korelasyonu test eder."""
    onnx_fp32 = os.path.join(gecici_dizin, "num_fp32.onnx")
    onnx_int8 = os.path.join(gecici_dizin, "num_int8.onnx")

    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(ornek_pytorch_model, torch.randn(1, 3, 32, 32), onnx_fp32)

    kuantizator = INT8Kuantizator()
    kuantizator.dinamik_kuantize_et(onnx_fp32, onnx_int8)

    karsilastirici = ModelBenchmarkKarsilastirici(
        pytorch_model=ornek_pytorch_model,
        onnx_fp32_yolu=onnx_fp32,
        onnx_int8_yolu=onnx_int8
    )

    test_girdi = np.random.randn(2, 3, 32, 32).astype(np.float32)
    esdegerlik = karsilastirici.sayisal_esdegerlik_test_et(test_girdi)

    # FP32 eşdeğerliği neredeyse kusursuz olmalıdır
    assert esdegerlik["fp32_kosinus_benzerligi"] > 0.999
    # INT8 kuantizasyon benzerliği yüksek olmalıdır
    assert esdegerlik["int8_kosinus_benzerligi"] > 0.90


def test_model_benchmark_karsilastirici(ornek_pytorch_model: UretimVisionNet, gecici_dizin: str):
    """Tam benchmark döngüsünün tüm metrikleri doğru ürettiğini test eder."""
    onnx_fp32 = os.path.join(gecici_dizin, "bench_fp32.onnx")
    onnx_int8 = os.path.join(gecici_dizin, "bench_int8.onnx")

    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(ornek_pytorch_model, torch.randn(1, 3, 32, 32), onnx_fp32)
    INT8Kuantizator().dinamik_kuantize_et(onnx_fp32, onnx_int8)

    karsilastirici = ModelBenchmarkKarsilastirici(
        pytorch_model=ornek_pytorch_model,
        onnx_fp32_yolu=onnx_fp32,
        onnx_int8_yolu=onnx_int8
    )

    girdi = np.random.randn(1, 3, 32, 32).astype(np.float32)
    sonuclar = karsilastirici.tam_benchmark_kos(girdi, tekrar=20)

    assert "pytorch_fp32" in sonuclar
    assert "onnx_fp32" in sonuclar
    assert "onnx_int8" in sonuclar
    assert sonuclar["onnx_fp32"]["speedup"] > 0


def test_gorsellestirici_paneli(ornek_pytorch_model: UretimVisionNet, gecici_dizin: str):
    """6 Panelli Capstone Teşhis Panosunun çizilip kaydedildiğini test eder."""
    onnx_fp32 = os.path.join(gecici_dizin, "gorsel_fp32.onnx")
    onnx_int8 = os.path.join(gecici_dizin, "gorsel_int8.onnx")
    panel_yolu = os.path.join(gecici_dizin, "panel.png")

    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(ornek_pytorch_model, torch.randn(1, 3, 32, 32), onnx_fp32)
    INT8Kuantizator().dinamik_kuantize_et(onnx_fp32, onnx_int8)

    karsilastirici = ModelBenchmarkKarsilastirici(
        pytorch_model=ornek_pytorch_model,
        onnx_fp32_yolu=onnx_fp32,
        onnx_int8_yolu=onnx_int8
    )

    girdi = np.random.randn(1, 3, 32, 32).astype(np.float32)
    esdegerlik = karsilastirici.sayisal_esdegerlik_test_et(girdi)
    bench = karsilastirici.tam_benchmark_kos(girdi, tekrar=10)

    cizim_yolu = CapstoneGorsellestirici.panoyu_ciz_ve_kaydet(
        benchmark_sonuclari=bench,
        esdegerlik_sonuclari=esdegerlik,
        cikti_yolu=panel_yolu
    )

    assert os.path.exists(cizim_yolu)
    assert os.path.getsize(cizim_yolu) > 10000
