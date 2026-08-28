"""
Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru Birim Testleri
-------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import pytest
import torch

from src.model import VisionClassifier
from src.dinamik_batcher import DinamikBatchMotoru, CikarimYaniti
from src.benchmark_motoru import BatchingBenchmarkMotoru


@pytest.fixture
def dummy_model():
    return VisionClassifier(giris_kanali=3, sinif_sayisi=10, taban_kanal=8)


def test_dinamik_batcher_baslatma_ve_kapatma(dummy_model):
    motor = DinamikBatchMotoru(model=dummy_model, max_batch_size=8, max_bekleme_ms=10.0, cihaz="cpu")
    assert motor.calisiyor is True
    motor.kapat()
    assert motor.calisiyor is False


def test_tekil_istek_senkron_tahmin(dummy_model):
    motor = DinamikBatchMotoru(model=dummy_model, max_batch_size=8, max_bekleme_ms=10.0, cihaz="cpu")
    girdi = torch.randn(1, 3, 32, 32)

    yanit = motor.tahmin_et_senkron(girdi)
    assert isinstance(yanit, CikarimYaniti)
    assert yanit.cikis.shape == (1, 10)
    assert yanit.toplam_gecikme_ms > 0
    assert yanit.batch_boyutu == 1
    motor.kapat()


def test_coklu_asenkron_istek_batch_birlestirme(dummy_model):
    motor = DinamikBatchMotoru(model=dummy_model, max_batch_size=8, max_bekleme_ms=50.0, cihaz="cpu")
    gelecekler = []

    # 4 isteği aynı anda gönder
    for i in range(4):
        g = torch.randn(3, 32, 32)
        gelecekler.append(motor.tahmin_et_asenkron(g, istek_id=f"req_{i}"))

    sonuclar = [f.result(timeout=2.0) for f in gelecekler]
    assert len(sonuclar) == 4
    for r in sonuclar:
        assert r.cikis.shape == (1, 10)
        assert r.batch_boyutu >= 2  # Birleşmiş olmalı
    motor.kapat()


def test_max_batch_size_siniri(dummy_model):
    motor = DinamikBatchMotoru(model=dummy_model, max_batch_size=4, max_bekleme_ms=100.0, cihaz="cpu")
    gelecekler = []

    # 10 istek gönder
    for i in range(10):
        g = torch.randn(3, 32, 32)
        gelecekler.append(motor.tahmin_et_asenkron(g))

    sonuclar = [f.result(timeout=3.0) for f in gelecekler]
    assert len(sonuclar) == 10
    for r in sonuclar:
        assert r.batch_boyutu <= 4  # Asla max_batch_size (4) aşılmamalı
    motor.kapat()


def test_zaman_asimi_tetikleme(dummy_model):
    motor = DinamikBatchMotoru(model=dummy_model, max_batch_size=32, max_bekleme_ms=15.0, cihaz="cpu")
    girdi = torch.randn(3, 32, 32)

    t0 = time.time()
    yanit = motor.tahmin_et_senkron(girdi)
    gecen_sure_ms = (time.time() - t0) * 1000.0

    assert yanit.batch_boyutu == 1
    assert gecen_sure_ms >= 10.0  # Zaman aşımı kadar bekleyip batch tetiklendi
    motor.kapat()


def test_gecersiz_girdi_hatasi(dummy_model):
    motor = DinamikBatchMotoru(model=dummy_model, max_batch_size=8, max_bekleme_ms=10.0, cihaz="cpu")
    gecersiz_girdi = torch.randn(5, 3, 32, 32)  # Batch size > 1 olamaz

    with pytest.raises(ValueError):
        _ = motor.tahmin_et_asenkron(gecersiz_girdi)
    motor.kapat()


def test_benchmark_ardisik_b1(dummy_model):
    bench = BatchingBenchmarkMotoru(model=dummy_model, cihaz="cpu")
    istekler = [torch.randn(3, 32, 32) for _ in range(5)]

    sonuc = bench.kos_ardisik_b1(istekler)
    assert sonuc["toplam_istek"] == 5
    assert sonuc["throughput_req_s"] > 0
    assert sonuc["p50_gecikme_ms"] > 0


def test_benchmark_dinamik_batching(dummy_model):
    bench = BatchingBenchmarkMotoru(model=dummy_model, cihaz="cpu")
    istekler = [torch.randn(3, 32, 32) for _ in range(8)]

    sonuc = bench.kos_dinamik_batching(istekler, max_batch_size=4, max_bekleme_ms=10.0, es_zamanli_istemci_sayisi=4)
    assert sonuc["toplam_istek"] == 8
    assert sonuc["throughput_req_s"] > 0
    assert "ortalama_batch" in sonuc
