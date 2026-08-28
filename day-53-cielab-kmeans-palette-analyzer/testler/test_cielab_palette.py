"""
Day 53: CIELAB Renk Uzayında K-Means & Delta-E 2000 Hassas Tolerans Analizi Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.renk_uzayi_donusturucu import RenkUzayiDonusturucu
from src.cielab_kmeans_analizor import CIELABKMeansPaletAnalizoru
from src.delta_e_hesaplayici import DeltaEHesaplayici
from src.gorsellestirici import PaletAnalizGorsellestirici


def test_renk_donusumu_rgb_cielab_dongusu():
    """RGB -> CIELAB -> RGB tam döngüsünün 1 piksel toleransla korunduğunu test eder."""
    orijinal_rgb = np.array([[[120, 45, 200], [10, 220, 140]]], dtype=np.uint8)
    lab = RenkUzayiDonusturucu.rgb_to_cielab(orijinal_rgb)
    geri_rgb = RenkUzayiDonusturucu.cielab_to_rgb(lab)

    fark = np.max(np.abs(orijinal_rgb.astype(int) - geri_rgb.astype(int)))
    assert fark <= 1


def test_hex_rgb_donusumu():
    """HEX ve RGB dönüşümlerinin doğruluğunu test eder."""
    hex_kod = "#FF5733"
    rgb = RenkUzayiDonusturucu.hex_to_rgb(hex_kod)
    assert tuple(rgb) == (255, 87, 51)

    yeni_hex = RenkUzayiDonusturucu.rgb_to_hex(rgb)
    assert yeni_hex == hex_kod


def test_delta_e_76_ve_2000_ozdes_renk():
    """Aynı iki renk arasında Delta-E 76 ve CIEDE2000 değerlerinin 0.0 olduğunu test eder."""
    lab1 = [50.0, 20.0, -10.0]
    lab2 = [50.0, 20.0, -10.0]

    de76 = DeltaEHesaplayici.delta_e_76(lab1, lab2)
    de00 = DeltaEHesaplayici.delta_e_2000(lab1, lab2)

    assert de76 == 0.0
    assert de00 == 0.0


def test_ciede2000_mavi_donme_ve_kroma():
    """CIEDE2000 algoritmasının mavi bölgedeki dönme ve kroma düzeltmelerini başarıyla işlediğini test eder."""
    # Mavi bölgede hafif ton kayması
    lab1 = [50.0, 5.0, -40.0]
    lab2 = [50.0, 8.0, -42.0]

    de76 = DeltaEHesaplayici.delta_e_76(lab1, lab2)
    de00 = DeltaEHesaplayici.delta_e_2000(lab1, lab2)

    assert de76 > 0.0
    assert de00 > 0.0
    assert abs(de76 - de00) < 5.0  # Mantıklı korelasyon aralığı


def test_cielab_kmeans_palet_orani():
    """K-Means ile çıkarılan palet oranlarının toplamının %100 olduğunu test eder."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    analizor = CIELABKMeansPaletAnalizoru(k_renk=4, random_state=42)
    sonuc = analizor.palet_cikar(img)

    assert len(sonuc["palet"]) == 4
    toplam_oran = sum(p["yuzde"] for p in sonuc["palet"])
    assert pytest.approx(toplam_oran, 0.1) == 100.0


def test_tolerans_siniflandirma_kriterleri():
    """Delta-E 2000 tolerans motorunun eşikleri doğru sınıflandırdığını test eder."""
    t1 = DeltaEHesaplayici.tolerans_degerlendir(0.6)
    t2 = DeltaEHesaplayici.tolerans_degerlendir(1.5)
    t3 = DeltaEHesaplayici.tolerans_degerlendir(3.2)
    t4 = DeltaEHesaplayici.tolerans_degerlendir(6.8)

    assert t1["kod"] == "MUKEMMEL_ESLESME" and t1["seviye"] == "PASS"
    assert t2["kod"] == "TOLERANS_DAHILINDE" and t2["seviye"] == "PASS"
    assert t3["kod"] == "KABUL_SINIRINDA" and t3["seviye"] == "WARNING"
    assert t4["kod"] == "KRITIK_RED" and t4["seviye"] == "REJECT"


def test_gorsellestirici_panel_cizimi(tmp_path):
    """6 panelli teşhis panosunun geçerli bir PNG ürettiğini test eder."""
    img1 = np.full((60, 60, 3), 100, dtype=np.uint8)
    img2 = np.full((60, 60, 3), 110, dtype=np.uint8)

    analizor = CIELABKMeansPaletAnalizoru(k_renk=2, random_state=42)
    res1 = analizor.palet_cikar(img1)
    res2 = analizor.palet_cikar(img2)

    karsilastirma = [{
        "sira": 1,
        "ref_hex": "#646464",
        "num_hex": "#6E6E6E",
        "delta_e_76": 2.5,
        "delta_e_2000": 1.8,
        "tolerans": DeltaEHesaplayici.tolerans_degerlendir(1.8)
    }]

    cikis_yolu = str(tmp_path / "test_palet_paneli.png")
    yol = PaletAnalizGorsellestirici.panel_ciz(
        hedef_gorsel=img1,
        numune_gorsel=img2,
        hedef_analiz=res1,
        numune_analiz=res2,
        karsilastirma_sonuclari=karsilastirma,
        ortalama_de00=1.8,
        tolerans_ozet=DeltaEHesaplayici.tolerans_degerlendir(1.8),
        hedef_path=cikis_yolu
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
