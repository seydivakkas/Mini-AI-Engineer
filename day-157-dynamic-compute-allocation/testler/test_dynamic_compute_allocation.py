"""
GÜN 157: Dynamic Compute Allocation Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.zorluk_tahmincisi import ZorlukTahmincisi
from src.dinamik_butce_yoneticisi import DinamikButceYoneticisi
from src.cikarim_simulasyonu import CikarimSimulasyonu
from src.gorsellestirici import DinamikComputeGorsellestirici


def test_zorluk_tahmincisi_kolay():
    """Triviyal bilgi sorgusunun Kolay olarak sınıflandırıldığını test eder."""
    soru = "Türkiye'nin başkenti neresidir?"
    sonuc = ZorlukTahmincisi.zorluk_hesapla(soru)

    assert sonuc["kategori"] == "Kolay"
    assert sonuc["zorluk_skoru"] <= 0.30


def test_zorluk_tahmincisi_orta():
    """Aritmetik hesaplama sorusunun Orta olarak sınıflandırıldığını test eder."""
    soru = "Ahmet 250 TL'lik ceketi %20 indirimle kaça alır? Adım adım hesapla."
    sonuc = ZorlukTahmincisi.zorluk_hesapla(soru)

    assert sonuc["kategori"] == "Orta"
    assert 0.30 < sonuc["zorluk_skoru"] < 0.70


def test_zorluk_tahmincisi_zor():
    """Teorem ve algoritma optimizasyon sorusunun Zor olarak sınıflandırıldığını test eder."""
    soru = "Dinamik programlama ile TSP algoritmasını optimize edip teorem ile ispatlayınız."
    sonuc = ZorlukTahmincisi.zorluk_hesapla(soru)

    assert sonuc["kategori"] == "Zor"
    assert sonuc["zorluk_skoru"] >= 0.70


def test_dinamik_butce_tahsisi_kolay():
    """Kolay soruya 32 token ve doğrudan yanıt tahsis edildiğini test eder."""
    tahsis = DinamikButceYoneticisi.butce_tahsis_et("Python nedir?")

    assert tahsis["kategori"] == "Kolay"
    assert tahsis["tahsis_edilen_token_butcesi"] == 32
    assert tahsis["arama_derinligi"] == 0


def test_dinamik_butce_tahsisi_zor():
    """Zor soruya 4096 token ve MCTS derin arama tahsis edildiğini test eder."""
    tahsis = DinamikButceYoneticisi.butce_tahsis_et("AIME teorem ispatı ve algoritma optimizasyonu")

    assert tahsis["kategori"] == "Zor"
    assert tahsis["tahsis_edilen_token_butcesi"] == 4096
    assert tahsis["arama_derinligi"] == 8


def test_cikarim_simulasyonu_tasarruf():
    """Dinamik tahsisin sabit bütçeye göre %50'den fazla token ve maliyet tasarrufu sağladığını test eder."""
    sorular = [
        "Fransa'nın başkenti neresidir?",
        "20 ile 30'u çarpıp yüzde 18 ekle ve hesapla",
        "AIME teorem ispatı ve algoritma optimizasyonu",
    ]
    sonuc = CikarimSimulasyonu.calistir(sorular)

    assert sonuc["token_tasarrufu_yuzde"] > 50.0
    assert sonuc["maliyet_tasarrufu_yuzde"] > 50.0


def test_cikarim_simulasyonu_hizlanma():
    """Dinamik çıkarımın ortalama 2.0 kattan fazla hızlanma sağladığını test eder."""
    sorular = [
        "Fransa'nın başkenti neresidir?",
        "Python nedir?",
        "AIME teorem ispatı ve algoritma optimizasyonu",
    ]
    sonuc = CikarimSimulasyonu.calistir(sorular)

    assert sonuc["hizlanma_orani"] >= 2.0


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun PNG olarak kaydedildiğini test eder."""
    sorular = ["Soru 1", "Soru 2 hesapla", "Soru 3 teorem ispat"]
    sonuc = CikarimSimulasyonu.calistir(sorular)
    gorsellestirici = DinamikComputeGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_dynamic_compute.png")
        gorsellestirici.pano_olustur(sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
