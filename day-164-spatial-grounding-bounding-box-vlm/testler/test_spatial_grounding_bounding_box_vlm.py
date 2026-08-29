"""
GÜN 164: Spatial Grounding Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.koordinat_ayristirici import KoordinatAyristirici
from src.iou_degerlendirici import IoUDegerlendirici
from src.grounded_vlm_motoru import GroundedVLMMotoru
from src.gorsellestirici import SpatialGroundingGorsellestirici


def test_koordinat_ayristirici_tek_kutu():
    """Metin içindeki tekli [ymin, xmin, ymax, xmax] koordinatını doğru çıkardığını test eder."""
    metin = "Araba konumu: [100, 150, 500, 600]"
    kutular = KoordinatAyristirici.metinden_koordinat_cikar(metin)

    assert len(kutular) == 1
    assert kutular[0] == [100, 150, 500, 600]


def test_koordinat_ayristirici_coklu_kutu():
    """Metin içindeki çoklu kutuları sırasıyla çıkardığını test eder."""
    metin = "Kedi [100, 100, 300, 300] ve köpek [400, 500, 800, 900]"
    kutular = KoordinatAyristirici.metinden_koordinat_cikar(metin)

    assert len(kutular) == 2
    assert kutular[0] == [100, 100, 300, 300]
    assert kutular[1] == [400, 500, 800, 900]


def test_piksel_koordinatina_donustur():
    """0-1000 normalize koordinatların doğru piksel değerlerine dönüştürüldüğünü test eder."""
    norm_kutu = [500, 500, 1000, 1000]  # Sağ alt çeyrek
    px = KoordinatAyristirici.piksel_koordinatina_donustur(norm_kutu, resim_genislik=640, resim_yukseklik=480)

    assert px == [240, 320, 480, 640]


def test_iou_mukemmel_eslesme():
    """Tamamen örtüşen iki kutuda IoU'nun 1.0 olduğunu test eder."""
    k1 = [100, 100, 500, 500]
    k2 = [100, 100, 500, 500]
    assert IoUDegerlendirici.iou_hesapla(k1, k2) == 1.0


def test_iou_sifir_kesisim():
    """Hiç kesişmeyen iki kutuda IoU'nun 0.0 olduğunu test eder."""
    k1 = [0, 0, 100, 100]
    k2 = [500, 500, 600, 600]
    assert IoUDegerlendirici.iou_hesapla(k1, k2) == 0.0


def test_iou_kismi_kesisim():
    """Kısmi kesişimde IoU'nun doğru hesaplandığını test eder."""
    k1 = [0, 0, 100, 100]       # Alan = 10000
    k2 = [50, 50, 150, 150]     # Alan = 10000
    # Kesişim: [50, 50, 100, 100] -> Alan = 50*50 = 2500
    # Birleşim: 10000 + 10000 - 2500 = 17500
    # IoU = 2500 / 17500 = 1/7 = 0.142857
    iou = IoUDegerlendirici.iou_hesapla(k1, k2)
    assert abs(iou - 0.142857) < 1e-4


def test_grounded_vlm_motoru_senaryolar():
    """Grounded VLM motorunun senaryoları başarıyla işlediğini test eder."""
    rapor = GroundedVLMMotoru.senaryolari_degerlendir()

    assert len(rapor["senaryo_sonuclari"]) == 4
    assert rapor["genel_ozet"]["map_50_yuzdesi"] == 100.0
    assert rapor["genel_ozet"]["ortalama_iou"] > 0.80


def test_gorsellestirici_pano_uretme():
    """6 panelli spatial grounding teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = GroundedVLMMotoru.senaryolari_degerlendir()
    gorsellestirici = SpatialGroundingGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_grounding_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
