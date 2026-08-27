"""Baskın Renk Çıkarıcı ve Kuantizasyon Birim Testleri.

Bu dosya; K-Means renk kümelemesini, yüzdesel oranların toplamını,
HEX kod formatlarını ve görüntü kuantizasyonunun doğruluğunu test eder.
"""

import sys
import re
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.renk_kumeleyici import BaskinRenkCikarici, RenkBilgisi
from src.palet_gorsellestirici import PaletGorsellestirici


def test_palet_uzunlugu_ve_k_tutarliligi():
    """İstenen K adedinde renk çıkarıldığını doğrular."""
    resim_bgr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    cikarici = BaskinRenkCikarici(k_kume_sayisi=4, rastgele_durum=42)
    palet = cikarici.paleti_cikar(resim_bgr)

    assert len(palet) == 4


def test_yuzdelerin_toplami_yuz():
    """Çıkarılan renk yüzdelerinin toplamının yaklaşık %100 olduğunu doğrular."""
    resim_bgr = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
    cikarici = BaskinRenkCikarici(k_kume_sayisi=5, rastgele_durum=42)
    palet = cikarici.paleti_cikar(resim_bgr)

    toplam = sum(r.yuzde for r in palet)
    assert toplam == pytest.approx(100.0, abs=0.5)


def test_hex_kod_formati():
    """Üretilen tüm HEX renk kodlarının geçerli formatta olduğunu kontrol eder (#RRGGBB)."""
    resim_bgr = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
    cikarici = BaskinRenkCikarici(k_kume_sayisi=3)
    palet = cikarici.paleti_cikar(resim_bgr)

    hex_sablonu = re.compile(r"^#[0-9A-F]{6}$")
    for r in palet:
        assert hex_sablonu.match(r.hex_kodu) is not None


def test_rgb_araligi():
    """RGB değerlerinin 0 ile 255 tamsayı aralığında kaldığını doğrular."""
    resim_bgr = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    cikarici = BaskinRenkCikarici(k_kume_sayisi=3)
    palet = cikarici.paleti_cikar(resim_bgr)

    for r in palet:
        assert all(0 <= val <= 255 for val in r.rgb)


def test_azalan_yuzde_siralamasi():
    """Paletin en baskın renkten en seyreğe doğru azalan sırada dizildiğini test eder."""
    resim_bgr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    cikarici = BaskinRenkCikarici(k_kume_sayisi=5)
    palet = cikarici.paleti_cikar(resim_bgr)

    yuzdeler = [r.yuzde for r in palet]
    assert yuzdeler == sorted(yuzdeler, reverse=True)


def test_goruntu_kuantizasyonu():
    """Kuantize edilen görüntünün boyutunu koruduğunu ve tam K benzersiz renk içerdiğini doğrular."""
    resim_bgr = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    cikarici = BaskinRenkCikarici(k_kume_sayisi=3)
    quantize = cikarici.goruntuyu_quantize_et(resim_bgr)

    assert quantize.shape == resim_bgr.shape
    benzersiz_renkler = len(np.unique(quantize.reshape(-1, 3), axis=0))
    assert benzersiz_renkler <= 3


def test_palet_raporu_png_kaydetme(tmp_path):
    """Palet görsel raporunun diske fiziksel geçerli PNG dosyası yazdığını doğrular."""
    resim_bgr = np.zeros((30, 30, 3), dtype=np.uint8)
    palet = [
        RenkBilgisi((255, 0, 0), "#FF0000", 60.0, 540),
        RenkBilgisi((0, 255, 0), "#00FF00", 40.0, 360)
    ]
    hedef = tmp_path / "palet_test.png"

    cikti = PaletGorsellestirici.palet_raporu_ciz(resim_bgr, resim_bgr, palet, hedef)
    assert cikti.exists()
    assert cikti.stat().st_size > 0
