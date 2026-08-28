"""
Day 51: Pillow ile Hataya Toleranslı ve Güvenli Görsel Yükleyici Birim Testleri.
"""

import io
import os
import pytest
import numpy as np
from PIL import Image
from src.guvenli_yukleyici import GuvenliGorselYukleyici
from src.anomali_denetleyici import GorselSaglikDenetleyicisi
from src.gorsellestirici import GuvenliYukleyiciGorsellestirici


@pytest.fixture
def yukleyici():
    return GuvenliGorselYukleyici(maks_piksel_limiti=1_000_000)  # 1 MP test limiti


def test_normal_rgb_yukleme(yukleyici):
    """Normal bir RGB görselin güvenle ve doğru boyutta yüklendiğini test eder."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    res = yukleyici.guvenli_yukle(buf.getvalue())
    assert res["durum"] == "BASARILI"
    assert res["son_mod"] == "RGB"
    assert res["gorsel_numpy"].shape == (100, 100, 3)


def test_rgba_alfa_mat_kompoziti(yukleyici):
    """RGBA görselin alfa kanalının temizlenip 3 kanallı RGB'ye dönüştürüldüğünü test eder."""
    img = Image.new("RGBA", (80, 80), color=(0, 255, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = yukleyici.guvenli_yukle(buf.getvalue())
    assert res["durum"] == "BASARILI"
    assert res["son_mod"] == "RGB"
    assert res["gorsel_numpy"].shape == (80, 80, 3)


def test_cmyk_donusumu(yukleyici):
    """CMYK görselin RGB renk uzayına başarıyla dönüştürüldüğünü test eder."""
    img = Image.new("CMYK", (60, 60), color=(100, 50, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    res = yukleyici.guvenli_yukle(buf.getvalue())
    assert res["durum"] == "BASARILI"
    assert res["son_mod"] == "RGB"
    assert res["gorsel_numpy"].shape == (60, 60, 3)


def test_exif_oryantasyon_transpoze(yukleyici):
    """EXIF rotasyon etiketinin işlenip görselin transpoze edildiğini test eder."""
    exif_data = Image.Exif()
    exif_data[0x0112] = 6  # 90 derece saat yönü
    img = Image.new("RGB", (50, 100), color=(0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_data)

    res = yukleyici.guvenli_yukle(buf.getvalue())
    assert res["durum"] == "BASARILI"
    # 50x100 olan görsel 90 derece dönünce 100x50 olmalıdır
    assert res["son_boyut"] == (100, 50)


def test_kesik_dosya_kurtarma(yukleyici):
    """Kesik/yarım kalmış görsel verisinin çökmeden kurtarıldığını test eder."""
    img = Image.new("RGB", (200, 200), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    # Dosyanın sonundaki EOI marker ve son tarama satırlarını kes
    kesik_veri = buf.getvalue()[:int(len(buf.getvalue()) * 0.70)]

    res = yukleyici.guvenli_yukle(kesik_veri)
    assert res["durum"] == "BASARILI"
    assert res["gorsel_numpy"].shape == (200, 200, 3)


def test_decompression_bomb_engeli(yukleyici):
    """Maksimum piksel limitini aşan devasa görsellerin bellek ayrılmadan engellendiğini test eder."""
    # 1500 x 1000 = 1.5 MP > 1.0 MP Limit
    img_buyuk = Image.new("RGB", (1500, 1000), color=(0, 0, 0))
    buf = io.BytesIO()
    img_buyuk.save(buf, format="JPEG")

    res = yukleyici.guvenli_yukle(buf.getvalue())
    assert res["durum"] == "HATA"
    assert res["hata_turu"] == "DECOMPRESSION_BOMB_ENGELENDI"


def test_saglik_denetleyicisi_ve_panel(yukleyici, tmp_path):
    """Görsel anomali denetleyicisini ve görselleştiriciyi test eder."""
    dizi = np.ones((50, 50, 3), dtype=np.uint8) * 128
    saglik = GorselSaglikDenetleyicisi.denetle(dizi)
    assert "saglikli_mi" in saglik
    assert saglik["en_boy_orani"] == 1.0

    cikis_yolu = str(tmp_path / "test_panel.png")
    yol = GuvenliYukleyiciGorsellestirici.panel_ciz(
        ozet_metrikler={"toplam_islenen": 1, "engellenen_bomb": 0, "exif_duzeltilen": 0, "kurtarilan_kesik": 0, "rgba_donusturulen": 0},
        exif_ornek=dizi,
        rgba_ornek=dizi,
        kesik_ornek=dizi,
        hedef_path=cikis_yolu
    )
    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
