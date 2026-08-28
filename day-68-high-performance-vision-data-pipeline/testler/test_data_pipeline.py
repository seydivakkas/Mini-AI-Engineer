"""
Day 68: Yüksek Performanslı Veri Boru Hattı Birim Test Paketi
============================================================
Albumentations C++ dönüşümleri, Sentetik Veri Seti, CUDA Prefetcher,
Boru Hattı Benchmark suite ve Görselleştirici modüllerinin testleri.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.veri_donusturucu import YuksekPerformansArtirici
from src.veri_seti import SentetikGorselVeriSeti
from src.cuda_prefetcher import CUDAPrefetcher
from src.boru_hatti_karsilastirici import BoruHattiKarsilastirici
from src.gorsellestirici import VeriBoruHattiGorsellestirici


@pytest.fixture
def gecici_dizin():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def ornek_gorsel_np():
    return np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_yuksek_performans_artirici_egitim_donusumu(ornek_gorsel_np: np.ndarray):
    """Eğitim artırma zincirinin doğru boyutta CHW FloatTensor ürettiğini test eder."""
    artirici = YuksekPerformansArtirici(hedef_boyut=(48, 48))
    tensor_cikis = artirici.egitim_donustur(ornek_gorsel_np)

    assert isinstance(tensor_cikis, torch.Tensor)
    assert tensor_cikis.shape == (3, 48, 48)
    assert tensor_cikis.dtype == torch.float32


def test_yuksek_performans_artirici_dogrulama_donusumu(ornek_gorsel_np: np.ndarray):
    """Doğrulama zincirinin doğru boyutta tensör ürettiğini test eder."""
    artirici = YuksekPerformansArtirici(hedef_boyut=(64, 64))
    tensor_cikis = artirici.dogrulama_donustur(ornek_gorsel_np)

    assert isinstance(tensor_cikis, torch.Tensor)
    assert tensor_cikis.shape == (3, 64, 64)


def test_gecersiz_girdi_hatasi():
    """NumPy dizisi dışındaki girdilerde TypeError fırlatıldığını test eder."""
    artirici = YuksekPerformansArtirici()
    with pytest.raises(TypeError):
        artirici.egitim_donustur([1, 2, 3])  # type: ignore


def test_sentetik_gorsel_veri_seti_erisim():
    """SentetikGorselVeriSeti sınıfının indeksleme ve boyut doğruluğunu test eder."""
    artirici = YuksekPerformansArtirici(hedef_boyut=(32, 32))
    ds = SentetikGorselVeriSeti(ornek_sayisi=100, gorsel_boyutu=(32, 32), sinif_sayisi=5, donusum=artirici.egitim_donustur)

    assert len(ds) == 100
    img_t, label = ds[0]
    assert img_t.shape == (3, 32, 32)
    assert 0 <= label < 5


def test_cuda_prefetcher_iterasyon():
    """CUDAPrefetcher'ın DataLoader üzerindeki tüm batch'leri eksiksiz teslim ettiğini test eder."""
    ds = SentetikGorselVeriSeti(ornek_sayisi=80, gorsel_boyutu=(32, 32))
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    prefetcher = CUDAPrefetcher(loader)

    toplam_ornek = 0
    batch_sayisi = 0
    for x, y in prefetcher:
        toplam_ornek += x.size(0)
        batch_sayisi += 1

    assert batch_sayisi == 5
    assert toplam_ornek == 80


def test_cuda_prefetcher_bosluk_ve_durma():
    """Prefetcher'ın veri bittiğinde StopIteration fırlattığını test eder."""
    ds = SentetikGorselVeriSeti(ornek_sayisi=10, gorsel_boyutu=(16, 16))
    loader = DataLoader(ds, batch_size=10, shuffle=False)
    prefetcher = CUDAPrefetcher(loader)

    it = iter(prefetcher)
    x1, y1 = next(it)
    assert x1.size(0) == 10
    with pytest.raises(StopIteration):
        next(it)


def test_boru_hatti_benchmark_metrikleri():
    """BoruHattiKarsilastirici'nin tüm boru hatları için geçerli FPS ve gecikme ürettiğini test eder."""
    sonuc = BoruHattiKarsilastirici.benchmark_kos(
        ornek_sayisi=100,
        batch_size=20,
        gorsel_boyutu=(32, 32),
        tekrar_sayisi=1
    )

    assert "torchvision" in sonuc
    assert "albumentations_cpu" in sonuc
    assert "albumentations_prefetcher" in sonuc
    assert sonuc["torchvision"]["fps"] > 0
    assert sonuc["albumentations_cpu"]["fps"] > 0
    assert sonuc["albumentations_prefetcher"]["fps"] > 0


def test_veri_boru_hatti_gorsellestirici(gecici_dizin: str):
    """6 Panelli veri boru hattı teşhis panosunun dosyaya kaydedildiğini test eder."""
    panel_yolu = os.path.join(gecici_dizin, "test_veri_paneli.png")
    benchmark_sonuc = BoruHattiKarsilastirici.benchmark_kos(
        ornek_sayisi=64,
        batch_size=32,
        gorsel_boyutu=(32, 32),
        tekrar_sayisi=1
    )

    ornek_gorseller = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
    ornek_basliklar = ["1", "2", "3", "4"]

    cizim_yolu = VeriBoruHattiGorsellestirici.panoyu_ciz_ve_kaydet(
        benchmark_sonuclari=benchmark_sonuc,
        ornek_gorseller=ornek_gorseller,
        ornek_basliklar=ornek_basliklar,
        cikti_yolu=panel_yolu
    )

    assert os.path.exists(cizim_yolu)
    assert os.path.getsize(cizim_yolu) > 10000
