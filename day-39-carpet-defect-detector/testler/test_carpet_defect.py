"""
Day 39: Halı Dokuma Hataları ve Kusur Tespiti Birim Testleri.
"""

import os
import pytest
import numpy as np
from PIL import Image
from src.anomali_tespitci import AnomaliTespitci
from src.morfolojik_filtre import MorfolojikKusurFiltresi
from src.kontur_analizci import KonturAnalizci
from src.kusur_siniflandirici import KusurSiniflandirici
from src.sentetik_kusur_uretici import SentetikKusurluHaliUretici
from src.gorsellestirici import HaliKusurGorsellestirici


def test_anomali_tespitci_kalinti():
    """Aynı referans ve test görselinde kalıntının 0 olduğunu doğrular."""
    img = Image.new("RGB", (60, 60), color=(150, 150, 150))
    tespitci = AnomaliTespitci()
    sonuc = tespitci.anomali_haritasi_cikar(img, referans_gorseli=img)

    assert np.max(sonuc["kalinti_haritasi"]) == pytest.approx(0.0, abs=1e-5)
    assert np.sum(sonuc["ham_maske"]) == 0


def test_anomali_tespitci_kusurlu():
    """Kusur enjekte edildiğinde anomali tespit edildiğini test eder."""
    ref = Image.new("RGB", (80, 80), color=(200, 200, 200))
    test = ref.copy()
    test_arr = np.array(test)
    test_arr[30:50, 30:50] = [20, 20, 20]  # Koyu leke
    test_img = Image.fromarray(test_arr)

    tespitci = AnomaliTespitci()
    sonuc = tespitci.anomali_haritasi_cikar(test_img, referans_gorseli=ref)

    assert np.max(sonuc["anomali_skor_haritasi"]) > 0.5
    assert np.sum(sonuc["ham_maske"]) > 0


def test_morfolojik_filtre_gurultu_eleme():
    """Ayrık 1 piksel gürültülerin açma (opening) ile elendiğini test eder."""
    maske = np.zeros((50, 50), dtype=np.uint8)
    maske[5, 5] = 1   # Ayrık gürültü pikseli
    maske[20:35, 20:35] = 1  # Gerçek kusur bloğu

    filtre = MorfolojikKusurFiltresi(acma_iter=1, kapama_iter=1)
    temiz = filtre.temizle_ve_birlestir(maske)

    assert temiz[5, 5] == 0
    assert np.sum(temiz[20:35, 20:35]) > 100


def test_kontur_analizci_geometri():
    """Kontur analizcisinin en-boy oranı ve dairesellik hesaplamasını test eder."""
    maske = np.zeros((100, 100), dtype=np.uint8)
    # İnce uzun dikdörtgen (AR yüksek)
    maske[20:25, 10:80] = 1

    analizci = KonturAnalizci(min_kusur_alani=20)
    kusurlar = analizci.analiz_et(maske)

    assert len(kusurlar) == 1
    k = kusurlar[0]
    assert k["en_boy_orani"] > 5.0
    assert k["dairesellik"] < 0.4


def test_kusur_siniflandirici_iplik_kopmasi():
    """Yüksek en-boy oranına sahip hatanın IPLIK_KOPMASI olarak sınıflandırıldığını test eder."""
    kusur = {
        "kusur_id": "TEST-01",
        "alan": 250,
        "en_boy_orani": 4.5,
        "dairesellik": 0.2,
        "doluluk": 0.9,
        "kutu": [10, 10, 80, 10]
    }
    siniflandirilmis = KusurSiniflandirici.kusuru_siniflandir(kusur)
    assert siniflandirilmis["kusur_turu"] == "IPLIK_KOPMASI"
    assert siniflandirilmis["siddet"] == "ORTA_KUSUR"


def test_kusur_siniflandirici_parti_karari():
    """Kritik kusur varlığında partinin reddedildiğini test eder."""
    kusurlar = [
        {"kusur_id": "D-1", "alan": 600, "siddet": "KRITIK", "kusur_turu": "DELIK_YIRTIK"}
    ]
    rapor = KusurSiniflandirici.parti_kalite_degerlendir(kusurlar)
    assert rapor["parti_kalite_karari"] == "PARTI_RED_HURDA"
    assert rapor["parti_onayi"] is False


def test_hali_kusur_gorsellestirici_cizim(tmp_path):
    """6 panelli görselleştiricinin başarıyla PNG ürettiğini test eder."""
    img = Image.new("RGB", (60, 60), color=(200, 200, 200))
    anomali = np.zeros((60, 60), dtype=np.float64)
    maske = np.zeros((60, 60), dtype=np.uint8)

    kusurlar = [{
        "kusur_id": "D-01", "kutu": [10, 10, 20, 20], "alan": 100,
        "en_boy_orani": 1.2, "dairesellik": 0.8, "kusur_turu": "YAG_BOYA_LEKESI", "siddet": "ORTA_KUSUR"
    }]
    rapor = KusurSiniflandirici.parti_kalite_degerlendir(kusurlar)

    cikis_path = str(tmp_path / "test_kusur_panel.png")
    yol = HaliKusurGorsellestirici.kusur_paneli_ciz(
        img, anomali, maske, kusurlar, rapor, hedef_path=cikis_path
    )
    assert os.path.exists(yol)
