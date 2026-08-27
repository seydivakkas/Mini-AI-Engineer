"""Algısal Renk Benzerliği ve Arama Motoru Birim Testleri.

Bu dosya; RGB-LAB dönüşümünü, CIE76 ve CIEDE2000 Delta-E metriklerini,
palet mesafesini, arama sıralamasını ve görselleştiricinin PNG çıktısını doğrular.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.delta_e_hesaplayici import DeltaEHesaplayici
from src.palet_eslestirici import PaletRengi, PaletBenzerlikMotoru
from src.katalog_arama import KatalogUrunu, RenkTabanliAramaMotoru
from src.gorsellestirici import AramaGorsellestirici


def test_rgb_to_lab_donusumu():
    """RGB renginin [0, 100] L* aralığında 3 boyutlu LAB vektörüne dönüştüğünü doğrular."""
    kirmizi_lab = DeltaEHesaplayici.rgb_to_lab((255, 0, 0))
    assert kirmizi_lab.shape == (3,)
    assert 0.0 <= kirmizi_lab[0] <= 100.0


def test_ayni_renk_delta_e_sifir():
    """Özdeş renkler arasında CIE76 ve CIEDE2000 Delta-E mesafesinin 0.0 olduğunu doğrular."""
    renk_lab = np.array([60.0, 25.0, -15.0])
    assert DeltaEHesaplayici.cie76_mesafesi(renk_lab, renk_lab) == pytest.approx(0.0, abs=1e-5)
    assert DeltaEHesaplayici.ciede2000_mesafesi(renk_lab, renk_lab) == pytest.approx(0.0, abs=1e-5)


def test_ciede2000_pozitif_ve_simetrik():
    """CIEDE2000 metriğinin simetrik ve pozitif olduğunu doğrular."""
    c1 = DeltaEHesaplayici.rgb_to_lab((200, 50, 30))
    c2 = DeltaEHesaplayici.rgb_to_lab((50, 180, 220))

    d1 = DeltaEHesaplayici.ciede2000_mesafesi(c1, c2)
    d2 = DeltaEHesaplayici.ciede2000_mesafesi(c2, c1)

    assert d1 > 0.0
    assert d1 == pytest.approx(d2, abs=1e-4)


def test_ayni_palet_yuzde_yuz_benzerlik():
    """Birebir aynı iki paletin benzerlik skorunun %100 ve Delta-E mesafesinin 0.0 olduğunu test eder."""
    palet = [
        PaletRengi(rgb=(200, 40, 40), agirlik=0.6),
        PaletRengi(rgb=(40, 200, 40), agirlik=0.4),
    ]
    motor = PaletBenzerlikMotoru(metrik="ciede2000")
    skor, mesafe, _ = motor.benzerlik_skoru_hesapla(palet, palet)

    assert skor == pytest.approx(100.0, abs=0.1)
    assert mesafe == pytest.approx(0.0, abs=0.1)


def test_tamamen_farkli_palet_dusuk_benzerlik():
    """Tamamen zıt (Mavi vs Kırmızı) iki paletin benzerlik skorunun düşük çıktığını doğrular."""
    mavi_palet = [PaletRengi(rgb=(10, 30, 220), agirlik=1.0)]
    kirmizi_palet = [PaletRengi(rgb=(220, 20, 10), agirlik=1.0)]

    motor = PaletBenzerlikMotoru(metrik="ciede2000", hassasiyet_sigma=25.0)
    skor, mesafe, _ = motor.benzerlik_skoru_hesapla(mavi_palet, kirmizi_palet)

    assert mesafe > 40.0
    assert skor < 30.0


def test_katalog_arama_siralamasi():
    """Katalog aramasının sonuçları benzerlik skoruna göre kesin azalan sırada döndürdüğünü doğrular."""
    u1 = KatalogUrunu(
        urun_id="U1", ad="Mavi Ürün", kategori="Test",
        gorsel_bgr=np.zeros((30, 30, 3), dtype=np.uint8),
        palet=[PaletRengi(rgb=(20, 40, 200), agirlik=1.0)]
    )
    u2 = KatalogUrunu(
        urun_id="U2", ad="Kırmızı Ürün", kategori="Test",
        gorsel_bgr=np.zeros((30, 30, 3), dtype=np.uint8),
        palet=[PaletRengi(rgb=(210, 40, 20), agirlik=1.0)]
    )

    arama = RenkTabanliAramaMotoru()
    arama.urunleri_toplu_ekle([u1, u2])

    # Kırmızı tonlarında sorgu
    sorgu = [PaletRengi(rgb=(200, 30, 20), agirlik=1.0)]
    sonuclar = arama.arama_yap(sorgu, en_iyi_k=2)

    assert len(sonuclar) == 2
    assert sonuclar[0].urun.urun_id == "U2"  # Kırmızı ürün 1. sırada çıkmalı
    assert sonuclar[0].benzerlik_skoru >= sonuclar[1].benzerlik_skoru


def test_arama_raporu_png_kaydetme(tmp_path):
    """Arama sonuç panelinin fiziksel olarak geçerli PNG dosyası yazdığını doğrular."""
    sorgu_gorsel = np.zeros((40, 40, 3), dtype=np.uint8)
    sorgu_palet = [PaletRengi(rgb=(200, 50, 50), agirlik=1.0)]

    u = KatalogUrunu(
        urun_id="U1", ad="Test Ürün", kategori="Test",
        gorsel_bgr=np.zeros((40, 40, 3), dtype=np.uint8),
        palet=[PaletRengi(rgb=(200, 50, 50), agirlik=1.0)]
    )
    motor = PaletBenzerlikMotoru()
    skor, mesafe, eslesmeler = motor.benzerlik_skoru_hesapla(sorgu_palet, u.palet)

    from src.katalog_arama import AramaSonucu
    sonuc = [AramaSonucu(urun=u, benzerlik_skoru=skor, delta_e_mesafesi=mesafe, eslesmeler=eslesmeler)]

    hedef = tmp_path / "arama_test.png"
    cikti = AramaGorsellestirici.arama_raporu_ciz(sorgu_gorsel, sorgu_palet, sonuc, hedef)

    assert cikti.exists()
    assert cikti.stat().st_size > 0
