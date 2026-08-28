"""
Day 54: Dijital Adli Bilişim, Error Level Analysis (ELA) ve Görsel Manipülasyon Tespiti Birim Testleri.
"""

import os
import io
import pytest
import cv2
import numpy as np
from PIL import Image
from src.ela_analizoru import ErrorLevelAnalizoru
from src.gurultu_adli_analizor import GurultuAdliAnalizoru
from src.adli_teftis_motoru import AdliTeftisMotoru
from src.gorsellestirici import AdliTeftisGorsellestirici


@pytest.fixture
def ornek_gorseller():
    # 1. Otantik Dengelenmiş JPEG Görsel (Doğal Gradyan Zemin)
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 150)
    xx, yy = np.meshgrid(x, x)
    taban = (np.sin(xx) * 40.0 + np.cos(yy) * 40.0 + 150.0).astype(np.uint8)
    taban_rgb = cv2.cvtColor(taban, cv2.COLOR_GRAY2RGB)

    tampon = io.BytesIO()
    Image.fromarray(taban_rgb).save(tampon, format="JPEG", quality=80)
    tampon.seek(0)
    otantik_jpeg = np.array(Image.open(tampon).convert("RGB"))

    # 2. Spliced / Manipüle Edilmiş Görsel (Yüksek Kontrastlı Sahte Şekil/Damga)
    manipule = otantik_jpeg.copy()
    cv2.circle(manipule, (75, 75), 25, (220, 30, 30), 4)  # Yüksek kontrastlı mühür
    cv2.putText(manipule, "STAMP", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 30, 30), 1)

    return otantik_jpeg, manipule


def test_ela_hesaplama_boyutlari(ornek_gorseller):
    """ELA çıktılarının girdi boyutlarıyla tam eşleştiğini test eder."""
    otantik, _ = ornek_gorseller
    ela_rgb, fark_gri, istatistik = ErrorLevelAnalizoru.ela_hesapla(otantik, kalite=90)

    assert ela_rgb.shape == otantik.shape
    assert fark_gri.shape == otantik.shape[:2]
    assert "ortalama_hata" in istatistik


def test_ela_ayni_kalitede_sikistirilmis_gorsel(ornek_gorseller):
    """Homojen JPEG sıkıştırmasına sahip görselde ortalama hatanın düşük olduğunu test eder."""
    otantik, _ = ornek_gorseller
    _, fark_gri, istatistik = ErrorLevelAnalizoru.ela_hesapla(otantik, kalite=85)

    assert istatistik["ortalama_hata"] < 10.0


def test_ela_splicing_tespiti(ornek_gorseller):
    """Yabancı nesne eklenmiş (spliced) görselde maksimum hatanın belirgin yükseldiğini test eder."""
    otantik, manipule = ornek_gorseller
    _, _, ist_otantik = ErrorLevelAnalizoru.ela_hesapla(otantik, kalite=90)
    _, _, ist_manipule = ErrorLevelAnalizoru.ela_hesapla(manipule, kalite=90)

    assert ist_manipule["maks_hata"] > ist_otantik["maks_hata"]


def test_gurultu_kalintisi_hesaplama(ornek_gorseller):
    """Sensör gürültü kalıntısı ve lokal varyans haritasının başarıyla çıkarıldığını test eder."""
    otantik, _ = ornek_gorseller
    kalinti, kalinti_norm = GurultuAdliAnalizoru.gurultu_kalintisi_hesapla(otantik, filtre_ksize=3)
    varyans_harita, cv_skor = GurultuAdliAnalizoru.lokal_gurultu_varyansi_haritasi(kalinti, blok_boyutu=16)

    assert kalinti.shape == otantik.shape[:2]
    assert varyans_harita.shape == otantik.shape[:2]
    assert cv_skor >= 0.0


def test_adli_teftis_manipule_gorsel_tespiti(ornek_gorseller):
    """Manipüle edilmiş görselin adli teftiş motoru tarafından tespit edildiğini test eder."""
    _, manipule = ornek_gorseller
    motor = AdliTeftisMotoru(ela_kalite=90, z_esigi=2.0)
    sonuc = motor.teftis_et(manipule)

    assert sonuc["manipulasyon_skoru"] > 30.0
    assert len(sonuc["supheli_bolgeler"]) >= 1
    assert sonuc["risk_seviyesi"] in ["WARNING", "CRITICAL_REJECT"]


def test_adli_teftis_otantik_gorsel_karari(ornek_gorseller):
    """Homojen otantik görselin düşük risk skoru aldığını test eder."""
    otantik, _ = ornek_gorseller
    motor = AdliTeftisMotoru(ela_kalite=85, z_esigi=3.0)
    sonuc = motor.teftis_et(otantik)

    assert sonuc["manipulasyon_skoru"] < 40.0


def test_gorsellestirici_panel_cizimi(ornek_gorseller, tmp_path):
    """6 panelli adli teftiş panosunun geçerli bir PNG ürettiğini test eder."""
    _, manipule = ornek_gorseller
    motor = AdliTeftisMotoru()
    sonuc = motor.teftis_et(manipule)

    cikis_yolu = str(tmp_path / "test_adli_paneli.png")
    yol = AdliTeftisGorsellestirici.panel_ciz(sonuc, hedef_path=cikis_yolu)

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
