"""
FAZ 6 BÜYÜK FİNALİ: LLM Benchmark & Evaluation Testleri (Day 120).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import math
import pytest

from src.hakem_motoru import LLMHakemMotoru
from src.elo_motoru import ChatbotArenaEloMotoru
from src.faz6_modeller_benchmark import Faz6BenchmarkArenasi
from src.gorsellestirici import ArenaGorsellestirici


def test_tekli_puanlama_rubrik():
    """LLMHakemMotoru tekli puanlamanın 1-10 arası çalıştığını ve zengin içeriğe yüksek puan verdiğini test eder."""
    hakem = LLMHakemMotoru(seed=42)
    basit = "Sıralama yapılabilir."
    gelismis = "<think>Adım 1: Zaman O(N) analizi</think> ```python\ndef qsort(): pass\n``` Sonuç ispatlandı."

    p_basit = hakem.tekli_puanla("Soru", basit)["puan"]
    p_gelismis = hakem.tekli_puanla("Soru", gelismis)["puan"]

    assert 1.0 <= p_basit <= 10.0
    assert 1.0 <= p_gelismis <= 10.0
    assert p_gelismis > p_basit


def test_ciftli_karsilastir_tek_yon():
    """ciftli_karsilastir_tek_yon fonksiyonunun +1, -1 veya 0 döndürdüğünü test eder."""
    hakem = LLMHakemMotoru(seed=42)
    y_a = "```python\ndef ispat(): pass\n``` <think>Adım 1</think>"
    y_b = "Basit yanıt."

    sonuc = hakem.ciftli_karsilastir_tek_yon("Soru", y_a, y_b)
    assert sonuc in [1, -1, 0]
    assert sonuc == 1  # y_a daha kaliteli


def test_swap_test_pozisyon_yanliligi():
    """swap_testli_karsilastir fonksiyonunun tutarlı karşılaştırma yaptığını test eder."""
    hakem = LLMHakemMotoru(seed=42)
    y_a = "Üstün kaliteli kod ve ispat. ```python\ndef foo(): pass\n``` <think>Mantık</think>"
    y_b = "Kısa ve yüzeysel yanıt."

    karar, yanlilik_var_mi = hakem.swap_testli_karsilastir("Soru", y_a, y_b)
    assert karar == 1
    assert yanlilik_var_mi is False


def test_elo_beklenen_skor_matematigi():
    """Bradley-Terry beklenen skor formülünün (E_A + E_B = 1.0) matematiksel doğruluğunu test eder."""
    arena = ChatbotArenaEloMotoru(baslangic_elo=1000.0)
    e_a = arena.beklenen_skor(1200.0, 1000.0)
    e_b = arena.beklenen_skor(1000.0, 1200.0)

    assert math.isclose(e_a + e_b, 1.0, abs_tol=1e-5)
    assert e_a > e_b  # 1200 Elo'lu model 1000 Elo'lu modele karşı favoridir


def test_elo_mac_guncellemesi():
    """mac_isle fonksiyonunun kazananın Elo'sunu artırıp kaybedeninkini düşürdüğünü test eder."""
    arena = ChatbotArenaEloMotoru(baslangic_elo=1000.0, k_faktoru=32.0)
    arena.mac_isle("ModelA", "ModelB", sonuc=1)  # ModelA kazandı

    assert arena.elo_tablosu["ModelA"] > 1000.0
    assert arena.elo_tablosu["ModelB"] < 1000.0
    assert arena.mac_sayilari["ModelA"] == 1


def test_liderlik_tablosu_siralamasi():
    """liderlik_tablosu fonksiyonunun modelleri Elo puanına göre büyükten küçüğe sıraladığını test eder."""
    arena = ChatbotArenaEloMotoru(baslangic_elo=1000.0)
    arena.elo_tablosu["ZayıfModel"] = 800.0
    arena.elo_tablosu["GüçlüModel"] = 1200.0
    arena.mac_sayilari["ZayıfModel"] = 5
    arena.mac_sayilari["GüçlüModel"] = 5
    arena.galibiyet_sayilari["ZayıfModel"] = 1.0
    arena.galibiyet_sayilari["GüçlüModel"] = 4.0

    tablo = arena.liderlik_tablosu()
    assert tablo[0]["model_adi"] == "GüçlüModel"
    assert tablo[1]["model_adi"] == "ZayıfModel"
    assert tablo[0]["sira"] == 1


def test_faz6_benchmark_arenasi_turnuva():
    """Faz6BenchmarkArenasi sınıfının tüm modelleri yarıştırıp rapor ürettiğini test eder."""
    arena = Faz6BenchmarkArenasi(seed=42)
    rapor = arena.turnuvayi_kostur(mac_tur_sayisi=2)

    assert "liderlik_tablosu" in rapor
    assert len(rapor["liderlik_tablosu"]) == len(arena.MODELLER)
    assert "sampiyon_model" in rapor
    assert "mt_bench_kategori_skorlari" in rapor
    assert rapor["toplam_mac_sayisi"] > 0


def test_gorsellestirici_pano():
    """ArenaGorsellestirici sınıfının 6 panelli Capstone PNG teşhis dosyasını ürettiğini test eder."""
    arena = Faz6BenchmarkArenasi(seed=42)
    rapor = arena.turnuvayi_kostur(mac_tur_sayisi=1)

    gorsellestirici = ArenaGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_capstone_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
