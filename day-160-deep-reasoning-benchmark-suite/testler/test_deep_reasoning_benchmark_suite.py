"""
GÜN 160: Deep Reasoning Benchmark Suite Test Paketi (FAZ 8 BÜYÜK FİNALİ).
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.benchmark_veri_kumesi import BenchmarkVeriKumesi
from src.pass_at_k_degerlendirici import PassAtKDegerlendirici
from src.model_karsilastirici import ModelKarsilastirici
from src.gorsellestirici import FinalBenchmarkGorsellestirici


def test_benchmark_veri_kumesi_problemler():
    """Veri kümesinde AIME, GPQA ve ARC sorularının bulunduğunu test eder."""
    tum_sorular = BenchmarkVeriKumesi.problemleri_getir()

    assert len(tum_sorular) >= 6
    benchmarks = {p["benchmark"] for p in tum_sorular}
    assert "AIME" in benchmarks
    assert "GPQA Diamond" in benchmarks
    assert "ARC-Challenge" in benchmarks


def test_benchmark_veri_kumesi_filtreleme():
    """Belirli bir benchmark filtresinin doğru çalıştığını test eder."""
    aime_sorulari = BenchmarkVeriKumesi.problemleri_getir("AIME")

    assert len(aime_sorulari) == 2
    assert all(p["benchmark"] == "AIME" for p in aime_sorulari)


def test_pass_at_k_unbiased_formulu():
    """Unbiased Pass@k formülünün matematiksel doğruluğunu test eder."""
    # n=10, c=5, k=1 -> %50.0
    pass1 = PassAtKDegerlendirici.pass_at_k_hesapla(n=10, c=5, k=1)
    assert pass1 == 0.5

    # k=5 iken en az 1 doğru bulma ihtimali çok daha yüksektir (> 0.95)
    pass5 = PassAtKDegerlendirici.pass_at_k_hesapla(n=10, c=5, k=5)
    assert pass5 > 0.95


def test_pass_at_k_sinir_degerler():
    """Pass@k sınır durumlarını (c=0 ve c=n) test eder."""
    assert PassAtKDegerlendirici.pass_at_k_hesapla(n=10, c=0, k=2) == 0.0
    assert PassAtKDegerlendirici.pass_at_k_hesapla(n=10, c=10, k=2) == 1.0


def test_orneklem_degerlendir_majority_vote():
    """Örneklem sonuçları üzerinden çoğunluk oylamasının doğru çalıştığını test eder."""
    sonuclar = [True, True, True, False, False]
    degerlendirme = PassAtKDegerlendirici.orneklem_degerlendir(sonuclar, k_degerleri=[1, 2])

    assert degerlendirme["dogru_ornek_sayisi"] == 3
    assert degerlendirme["majority_vote_acc"] == 100.0


def test_model_karsilastirici_benchmark_sonuclari():
    """Model karşılaştırıcının 4 mimariyi de değerlendirdiğini test eder."""
    rapor = ModelKarsilastirici.benchmark_yurut()

    assert len(rapor["model_sonuclari"]) == 4
    assert rapor["faz8_toplam_kazanc_puani"] > 50.0


def test_model_karsilastirici_sampiyon():
    """DeepSeek-R1 Distill modelinin en yüksek DRI skoruna sahip olduğunu test eder."""
    rapor = ModelKarsilastirici.benchmark_yurut()

    assert "DeepSeek-R1" in rapor["sampiyon_model"]
    assert rapor["sampiyon_dri"] >= 85.0


def test_gorsellestirici_pano_uretme():
    """6 panelli FAZ 8 Büyük Final teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = ModelKarsilastirici.benchmark_yurut()
    gorsellestirici = FinalBenchmarkGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_final_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
