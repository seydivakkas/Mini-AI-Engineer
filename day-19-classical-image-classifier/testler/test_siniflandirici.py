"""Day 19 Birim Testleri: Geleneksel Makine Öğrenmesi ile Görsel Sınıflandırma."""

from pathlib import Path
import numpy as np
import pytest

from src.oznitelik_cikarici import KlasikOznitelikCikarici
from src.siniflandirici import GorselSiniflandirici, SiniflandiriciTipi, ModelSonucu
from src.degerlendirici import SiniflandirmaDegerlendirici


@pytest.fixture
def sentetik_veri():
    """Testler için 3 sınıflı sentetik öznitelik matrisi ve etiketler üretir."""
    np.random.seed(42)
    n_samples_per_class = 15
    dim = 94

    X_list = []
    y_list = []
    for sinif_id in range(3):
        merkez = np.random.randn(dim) * 2.0
        ornekler = merkez + np.random.randn(n_samples_per_class, dim) * 0.5
        X_list.append(ornekler)
        y_list.extend([sinif_id] * n_samples_per_class)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def test_oznitelik_cikarici_boyut():
    """Öznitelik çıkarıcının 94 boyutlu vektör ürettiğini doğrular."""
    cikarici = KlasikOznitelikCikarici(hedef_boyut=(64, 64))
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    vektor = cikarici.cikar(img)

    assert isinstance(vektor, np.ndarray)
    assert vektor.shape == (310,)
    assert not np.isnan(vektor).any()


def test_oznitelik_cikarici_gecersiz_girdi():
    """Boş veya 2 kanallı görselde hata fırlatıldığını doğrular."""
    cikarici = KlasikOznitelikCikarici()
    with pytest.raises(ValueError):
        cikarici.cikar(None)

    with pytest.raises(ValueError):
        cikarici.cikar(np.zeros((64, 64), dtype=np.uint8))


def test_svm_rbf_egitimi(sentetik_veri):
    """SVM (RBF) modelinin başarıyla eğitildiğini ve metrik ürettiğini doğrular."""
    X, y = sentetik_veri
    yonetici = GorselSiniflandirici(random_state=42)
    sonuc = yonetici.egit_ve_degerlendir(
        X[:30], y[:30], X[30:], y[30:], SiniflandiriciTipi.SVM_RBF, C=5.0
    )

    assert isinstance(sonuc, ModelSonucu)
    assert sonuc.accuracy >= 0.0
    assert sonuc.f1_macro >= 0.0
    assert len(sonuc.y_pred) == 15


def test_svm_linear_egitimi(sentetik_veri):
    """SVM (Linear) modelinin eğitildiğini ve tahmin ürettiğini doğrular."""
    X, y = sentetik_veri
    yonetici = GorselSiniflandirici(random_state=42)
    sonuc = yonetici.egit_ve_degerlendir(
        X[:30], y[:30], X[30:], y[30:], SiniflandiriciTipi.SVM_LINEAR, C=1.0
    )

    assert isinstance(sonuc, ModelSonucu)
    assert sonuc.accuracy >= 0.0
    assert len(sonuc.y_pred) == 15


def test_random_forest_egitimi_ve_onem(sentetik_veri):
    """Random Forest modelinin eğitilip özellik önem vektörü ürettiğini test eder."""
    X, y = sentetik_veri
    yonetici = GorselSiniflandirici(random_state=42)
    sonuc = yonetici.egit_ve_degerlendir(
        X[:30], y[:30], X[30:], y[30:], SiniflandiriciTipi.RANDOM_FOREST, n_estimators=50
    )

    assert isinstance(sonuc, ModelSonucu)
    assert sonuc.feature_importances is not None
    assert len(sonuc.feature_importances) == 94
    assert np.isclose(np.sum(sonuc.feature_importances), 1.0, atol=1e-3)


def test_capraz_dogrulama(sentetik_veri):
    """Stratified K-Fold çapraz doğrulamanın çalıştığını test eder."""
    X, y = sentetik_veri
    yonetici = GorselSiniflandirici(random_state=42)
    ort, std = yonetici.capraz_dogrulama_yap(X, y, SiniflandiriciTipi.SVM_RBF, k_kat=3)

    assert 0.0 <= ort <= 1.0
    assert std >= 0.0


def test_rapor_olusturma(sentetik_veri, tmp_path):
    """Değerlendirme raporunun PNG olarak kaydedildiğini test eder."""
    X, y = sentetik_veri
    yonetici = GorselSiniflandirici(random_state=42)

    sonuc1 = yonetici.egit_ve_degerlendir(
        X[:30], y[:30], X[30:], y[30:], SiniflandiriciTipi.SVM_RBF
    )
    sonuc2 = yonetici.egit_ve_degerlendir(
        X[:30], y[:30], X[30:], y[30:], SiniflandiriciTipi.RANDOM_FOREST, n_estimators=30
    )

    hedef_dosya = tmp_path / "test_raporu.png"
    cikti = SiniflandirmaDegerlendirici.kapsamli_rapor_olustur(
        sonuclar=[sonuc1, sonuc2],
        sinif_isimleri=["Sinif_A", "Sinif_B", "Sinif_C"],
        hedef_dosya=hedef_dosya,
    )

    assert cikti.exists()
    assert cikti.stat().st_size > 0


def test_en_iyi_hiperparametreleri_bul_svm(sentetik_veri):
    """GridSearch ile optimal SVM hiperparametrelerinin bulunduğunu test eder."""
    X, y = sentetik_veri
    yonetici = GorselSiniflandirici(random_state=42)

    param_grid = {
        "siniflandirici__C": [0.1, 1.0, 10.0],
        "siniflandirici__gamma": ["scale", "auto"],
    }
    en_iyi_params, en_iyi_skor, en_iyi_pipe = yonetici.en_iyi_hiperparametreleri_bul_svm(
        X, y, param_grid=param_grid, cv=3
    )

    assert "siniflandirici__C" in en_iyi_params
    assert "siniflandirici__gamma" in en_iyi_params
    assert 0.0 <= en_iyi_skor <= 1.0
    assert en_iyi_pipe is not None
