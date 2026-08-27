"""NumPy Görüntü Analizörü Birim Testleri.

Bu dosya, analizörün sınır durumlarını, istatistiksel hesaplama doğruluğunu,
tip güvenliklerini ve sayısal kararlılıklarını (taşma, sıfıra bölme) test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekler
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.goruntu_analizoru import NumPyGoruntuAnalizoru
from src.yardimcilar import sentetik_goruntu_uret


def test_gecersiz_girdi_hatalari():
    """Geçersiz tiplerde ve boyutlarda doğru hataların fırlatıldığını doğrular."""
    with pytest.raises(TypeError):
        NumPyGoruntuAnalizoru("gecersiz_tip")  # type: ignore

    with pytest.raises(ValueError):
        NumPyGoruntuAnalizoru(np.array([]))  # Boş dizi

    with pytest.raises(ValueError):
        NumPyGoruntuAnalizoru(np.zeros((10, 10, 10, 10)))  # 4B geçersiz


def test_sentetik_goruntu_boyut_ve_tipleri():
    """Sentetik üretim fonksiyonunun doğru boyut ve uint8 ürettiğini doğrular."""
    gorsel = sentetik_goruntu_uret(64, 32, desen_tipi="gradyan")
    assert gorsel.shape == (64, 32, 3)
    assert gorsel.dtype == np.uint8


def test_kanal_ayristirma():
    """RGB kanalların 2B matrislere doğru ayrıştığını kontrol eder."""
    gorsel = sentetik_goruntu_uret(32, 32, desen_tipi="renkli_bloklar")
    analizor = NumPyGoruntuAnalizoru(gorsel)
    kanallar = analizor.kanal_ayristir()

    assert set(kanallar.keys()) == {"Kirmizi", "Yesil", "Mavi"}
    for kanal_adi, matris in kanallar.items():
        assert matris.shape == (32, 32)
        assert matris.ndim == 2


def test_gri_ton_donusumu_matematigi():
    """BT.601 lüminans katsayılarının doğruluğunu test eder."""
    # Sadece saf kırmızı (255, 0, 0) pikselli bir görsel
    kirmizi_gorsel = np.zeros((10, 10, 3), dtype=np.uint8)
    kirmizi_gorsel[:, :, 0] = 255

    analizor = NumPyGoruntuAnalizoru(kirmizi_gorsel)
    gri = analizor.gri_tona_donustur()

    # Y = 0.299 * 255 = 76.245 -> yuvarlama ile 76 olmalıdır
    beklenen_deger = int(np.round(0.299 * 255))
    assert np.all(gri == beklenen_deger)


def test_min_max_normallestirme_araligi():
    """Min-Max dönüşümünün değerleri [0, 1] ve [-1, 1] sınırları içine aldığını doğrular."""
    gorsel = sentetik_goruntu_uret(40, 40, desen_tipi="gradyan")
    analizor = NumPyGoruntuAnalizoru(gorsel)

    norm_0_1 = analizor.min_max_normallestir(hedef_aralik=(0.0, 1.0))
    assert norm_0_1.min() >= 0.0
    assert norm_0_1.max() <= 1.0
    assert norm_0_1.dtype == np.float32

    norm_eksi1_1 = analizor.min_max_normallestir(hedef_aralik=(-1.0, 1.0))
    assert norm_eksi1_1.min() >= -1.0
    assert norm_eksi1_1.max() <= 1.0


def test_z_skoru_normallestirme_istatistigi():
    """Standartlaştırma sonrasında ortalamanın ~0, standart sapmanın ~1 olduğunu doğrular."""
    gorsel = sentetik_goruntu_uret(100, 100, desen_tipi="gradyan")
    analizor = NumPyGoruntuAnalizoru(gorsel)

    z_norm = analizor.z_skoru_normallestir(kanal_bazli=True)
    for k in range(3):
        assert np.isclose(float(np.mean(z_norm[:, :, k])), 0.0, atol=1e-3)
        assert np.isclose(float(np.std(z_norm[:, :, k])), 1.0, atol=1e-2)


def test_sayisal_tasma_engelleme():
    """uint8 parlaklık artırma sırasında taşma (modulo 256 sarma) olmadığını denetler."""
    matris = np.full((10, 10, 3), 200, dtype=np.uint8)
    analizor = NumPyGoruntuAnalizoru(matris)

    # 200 * 1.5 = 300 normalde uint8'de 44'e taşardı
    parlak = analizor.parlaklik_ayarla(1.5)
    assert parlak.dtype == np.uint8
    assert np.all(parlak == 255)


def test_guvenli_kirpma():
    """Görüntü sınırlarının dışına taşan kırpma koordinatlarının güvenle sınırlandığını doğrular."""
    gorsel = sentetik_goruntu_uret(50, 50, desen_tipi="gradyan")
    analizor = NumPyGoruntuAnalizoru(gorsel)

    kirpilmis = analizor.kirp(y_baslangic=-10, y_bitis=100, x_baslangic=10, x_bitis=30)
    assert kirpilmis.shape == (50, 20, 3)
