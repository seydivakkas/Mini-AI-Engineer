"""
Day 41: Uçtan Uca Çoklu Görev Halı Zekası Paketi Birim Testleri.
"""

import os
import pytest
import numpy as np
from PIL import Image
from src.moduller.renk_motoru import RenkZekasiMotoru
from src.moduller.arama_motoru import GorselAramaMotoru
from src.moduller.kusur_motoru import KusurTespitMotoru
from src.moduller.rag_motoru import SektorelRAGMotoru
from src.orkestrator import HaliZekasiOrkestrator
from src.gorsellestirici import HaliZekaPaketiGorsellestirici


def test_renk_motoru_cielab_kumeleme():
    """Renk motorunun iplik yüzdelerini %100 toplamla çıkardığını test eder."""
    img = Image.new("RGB", (60, 60), color=(140, 30, 50))
    motor = RenkZekasiMotoru()
    sonuc = motor.analiz_et(img, k_iplik=3)

    assert len(sonuc["iplikler"]) == 3
    toplam_yuzde = sum(i["yuzde"] for i in sonuc["iplikler"])
    assert toplam_yuzde == pytest.approx(100.0, abs=0.1)


def test_arama_motoru_top1_eslesme():
    """Görsel arama motorunun katalog eşleşmesi ürettiğini test eder."""
    motor = GorselAramaMotoru()
    img = Image.new("RGB", (60, 60), color=(138, 28, 48))
    sonuc = motor.ara(img, top_k=2)

    assert len(sonuc["top_sonuclar"]) == 2
    assert sonuc["en_iyi_eslesme"] is not None
    assert 0.0 <= sonuc["en_iyi_eslesme"]["benzerlik_skoru"] <= 100.0


def test_kusur_motoru_tespit():
    """Kusur motorunun anomali tespit edip sınıflandırdığını test eder."""
    ref = Image.new("RGB", (80, 80), color=(200, 200, 200))
    test = ref.copy()
    arr = np.array(test)
    arr[30:50, 30:50] = [10, 10, 10]
    test_img = Image.fromarray(arr)

    motor = KusurTespitMotoru()
    sonuc = motor.tespit_et(test_img, referans_gorseli=ref)

    assert sonuc["kusur_sayisi"] >= 1
    assert "IPLIK_KOPMASI" in [k["kusur_turu"] for k in sonuc["kusurlar"]] or "YAG_BOYA_LEKESI" in [k["kusur_turu"] for k in sonuc["kusurlar"]] or "DELIK_YIRTIK" in [k["kusur_turu"] for k in sonuc["kusurlar"]]


def test_rag_motoru_cozum_esleme():
    """RAG motorunun kusur türüne karşılık gelen standart çözümü getirdiğini test eder."""
    motor = SektorelRAGMotoru()
    cozum = motor.hata_icin_cozum_getir("IPLIK_KOPMASI")

    assert "standart_adi" in cozum
    assert "oneri" in cozum
    assert "çözgü" in cozum["oneri"].lower() or "büküm" in cozum["oneri"].lower() or "tansiyon" in cozum["oneri"].lower()


def test_orkestrator_tam_denetim():
    """Orkestratörün tüm modülleri konsolide edip eksiksiz rapor ürettiğini test eder."""
    img = Image.new("RGB", (100, 100), color=(220, 210, 195))
    orkestrator = HaliZekasiOrkestrator()
    rapor = orkestrator.tam_denetim_yap(img, k_iplik=3)

    assert "genel_kalite_skoru" in rapor
    assert "fabrika_karari" in rapor
    assert "renk_analizi" in rapor
    assert "gorsel_arama" in rapor
    assert "kusur_tespiti" in rapor
    assert "rag_cozum_onerileri" in rapor


def test_orkestrator_kusursuz_numune():
    """Kusursuz numunenin 1. Kalite onayı aldığını test eder."""
    ref = Image.new("RGB", (100, 100), color=(228, 217, 198))
    orkestrator = HaliZekasiOrkestrator()
    rapor = orkestrator.tam_denetim_yap(ref, referans_gorseli=ref, k_iplik=3)

    assert rapor["genel_kalite_skoru"] >= 90.0
    assert rapor["sevkiyat_onayi"] is True
    assert "1_KALITE" in rapor["fabrika_karari"]


def test_hali_zeka_paketi_gorsellestirici(tmp_path):
    """6 panelli yönetim panosunun başarıyla PNG ürettiğini test eder."""
    img = Image.new("RGB", (80, 80), color=(200, 200, 200))
    orkestrator = HaliZekasiOrkestrator()
    rapor = orkestrator.tam_denetim_yap(img, k_iplik=3)

    cikis_path = str(tmp_path / "test_suite_panel.png")
    yol = HaliZekaPaketiGorsellestirici.konsolide_panel_ciz(img, rapor, hedef_path=cikis_path)
    assert os.path.exists(yol)
