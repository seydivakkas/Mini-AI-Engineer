"""
Day 55: İleri PyTorch DataLoader, num_workers ve pin_memory Optimizasyonu Birim Testleri.
"""

import os
import pytest
import torch
from src.veri_seti_motoru import HizliSentetikGorselVeriSeti, worker_init_fn
from src.darbogaz_olcer import DataLoaderBenchmarkEngine
from src.gorsellestirici import DataLoaderAnalizGorsellestirici


@pytest.fixture
def ornek_veri_seti():
    return HizliSentetikGorselVeriSeti(
        num_samples=200,
        channels=3,
        height=32,
        width=32,
        num_classes=5,
        simule_io_ms=0.0,
        seed=42
    )


def test_hizli_veri_seti_uzunluk_ve_tipler(ornek_veri_seti):
    """Veri seti boyutlarının ve tensör tiplerinin doğru oluşturulduğunu test eder."""
    assert len(ornek_veri_seti) == 200
    x, y = ornek_veri_seti[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long


def test_worker_init_fn_tohumlama():
    """Worker tohumlama fonksiyonunun istisnasız çalıştığını test eder."""
    try:
        worker_init_fn(0)
        worker_init_fn(3)
    except Exception as e:
        pytest.fail(f"worker_init_fn istisna fırlattı: {e}")


def test_dataloader_tekil_olcum_yapisi(ornek_veri_seti):
    """Tekil benchmark ölçümünün tüm gerekli anahtarları ve pozitif işlem hızını döndüğünü test eder."""
    res = DataLoaderBenchmarkEngine.tekil_olcum(
        dataset=ornek_veri_seti,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        num_batches=5
    )

    assert "isleme_hizi_ornek_sn" in res
    assert "ort_batch_gecikmesi_ms" in res
    assert "toplam_sure_sn" in res
    assert res["isleme_hizi_ornek_sn"] > 0.0


def test_pin_memory_performansi(ornek_veri_seti):
    """pin_memory=True bayrağının DataLoader ile sorunsuz çalıştığını test eder."""
    res = DataLoaderBenchmarkEngine.tekil_olcum(
        dataset=ornek_veri_seti,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        num_batches=5
    )

    assert res["pin_memory"] is True
    assert res["toplam_ornek"] > 0


def test_karsilastirmali_benchmark_siralamasi(ornek_veri_seti):
    """Karşılaştırmalı benchmark motorunun 4 ana konfigürasyonu döndüğünü test eder."""
    sonuclar = DataLoaderBenchmarkEngine.karsilastirmali_benchmark(
        dataset=ornek_veri_seti,
        batch_size=32,
        num_batches=5
    )

    assert len(sonuclar) == 4
    for b in sonuclar:
        assert "hizlanma_carpani" in b
        assert "gpu_starvation_orani" in b


def test_worker_olceklenme_taramasi(ornek_veri_seti):
    """Worker tarama motorunun belirtilen tüm worker sayıları için ölçüm ürettiğini test eder."""
    workers = [0, 2]
    tarama = DataLoaderBenchmarkEngine.worker_olceklenme_taramasi(
        dataset=ornek_veri_seti,
        batch_size=32,
        worker_listesi=workers,
        num_batches=5
    )

    assert len(tarama) == 2
    assert tarama[0]["num_workers"] == 0
    assert tarama[1]["num_workers"] == 2


def test_gorsellestirici_panel_cizimi(ornek_veri_seti, tmp_path):
    """6 panelli teşhis panosunun başarıyla PNG dosyası ürettiğini test eder."""
    bench = DataLoaderBenchmarkEngine.karsilastirmali_benchmark(ornek_veri_seti, batch_size=32, num_batches=4)
    worker_tarama = DataLoaderBenchmarkEngine.worker_olceklenme_taramasi(ornek_veri_seti, batch_size=32, worker_listesi=[0, 1], num_batches=4)

    cikis_yolu = str(tmp_path / "test_dataloader_paneli.png")
    yol = DataLoaderAnalizGorsellestirici.panel_ciz(
        benchmark_sonuclari=bench,
        worker_tarama_sonuclari=worker_tarama,
        hedef_path=cikis_yolu
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
