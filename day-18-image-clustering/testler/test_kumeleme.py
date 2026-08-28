"""Day 18 Birim Testleri: Etiketsiz Görsellerin Otomatik Kümelenmesi."""

from pathlib import Path
import numpy as np
import pytest

from src.vektor_cikarici import GorselVektorCikarici
from src.kumeleme_motoru import GorselKumelemeMotoru, KumelemeSonucu
from src.gorsellestirici import KumeGorsellestirici


@pytest.fixture
def ornek_gorseller():
    """Testler için 4 farklı renkte sentetik test görselleri üretir."""
    gorseller = []
    # 4 Kırmızı görsel
    for _ in range(4):
        gorseller.append(np.full((32, 32, 3), (20, 20, 200), dtype=np.uint8))
    # 4 Mavi görsel
    for _ in range(4):
        gorseller.append(np.full((32, 32, 3), (200, 20, 20), dtype=np.uint8))
    # 4 Yeşil görsel
    for _ in range(4):
        gorseller.append(np.full((32, 32, 3), (20, 200, 20), dtype=np.uint8))
    return gorseller


def test_vektor_cikarici_boyut_ve_norm():
    """Öznitelik çıkarıcının 138 boyutlu ve L2 birim normlu vektör ürettiğini doğrular."""
    cikarici = GorselVektorCikarici(hedef_boyut=(32, 32))
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    vektor = cikarici.cikar(img)

    assert isinstance(vektor, np.ndarray)
    assert vektor.shape == (138,)
    norm = np.linalg.norm(vektor)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_vektor_cikarici_hatali_girdi():
    """Boş veya tek kanallı görsel verildiğinde ValueError fırlatıldığını test eder."""
    cikarici = GorselVektorCikarici()
    with pytest.raises(ValueError):
        cikarici.cikar(None)

    with pytest.raises(ValueError):
        cikarici.cikar(np.zeros((32, 32), dtype=np.uint8))  # 2D gri resim


def test_kmeans_kumeleme(ornek_gorseller):
    """K-Means algoritmasının doğru küme sayısı ve metrikler ürettiğini doğrular."""
    cikarici = GorselVektorCikarici(hedef_boyut=(32, 32))
    X = np.array([cikarici.cikar(img) for img in ornek_gorseller])

    motor = GorselKumelemeMotoru(random_state=42)
    sonuc = motor.k_means_kumele(X, n_clusters=3)

    assert isinstance(sonuc, KumelemeSonucu)
    assert sonuc.kume_sayisi == 3
    assert len(sonuc.etiketler) == 12
    assert sonuc.silhouette is not None
    assert sonuc.silhouette > 0.0


def test_kmeans_gecersiz_k():
    """Örnek sayısından büyük K seçildiğinde hata fırlatıldığını test eder."""
    motor = GorselKumelemeMotoru()
    X = np.random.randn(3, 10)
    with pytest.raises(ValueError):
        motor.k_means_kumele(X, n_clusters=5)


def test_dbscan_kumeleme(ornek_gorseller):
    """DBSCAN algoritmasının yoğunluk tabanlı kümeleri ayrıştırabildiğini test eder."""
    cikarici = GorselVektorCikarici(hedef_boyut=(32, 32))
    X = np.array([cikarici.cikar(img) for img in ornek_gorseller])

    motor = GorselKumelemeMotoru()
    sonuc = motor.dbscan_kumele(X, eps=0.4, min_samples=2)

    assert isinstance(sonuc, KumelemeSonucu)
    assert sonuc.kume_sayisi >= 1
    assert len(sonuc.etiketler) == len(X)


def test_agglomerative_kumeleme(ornek_gorseller):
    """Hiyerarşik (Agglomerative) kümelemenin başarıyla çalıştığını doğrular."""
    cikarici = GorselVektorCikarici(hedef_boyut=(32, 32))
    X = np.array([cikarici.cikar(img) for img in ornek_gorseller])

    motor = GorselKumelemeMotoru()
    sonuc = motor.agglomerative_kumele(X, n_clusters=3, metric="cosine", linkage="average")

    assert isinstance(sonuc, KumelemeSonucu)
    assert sonuc.kume_sayisi == 3
    assert len(set(sonuc.etiketler)) == 3


def test_en_iyi_k_bul_kmeans(ornek_gorseller):
    """Optimal K seçiminin en yüksek Silhouette skorlu K'yı seçtiğini test eder."""
    cikarici = GorselVektorCikarici(hedef_boyut=(32, 32))
    X = np.array([cikarici.cikar(img) for img in ornek_gorseller])

    motor = GorselKumelemeMotoru(random_state=42)
    en_iyi_k, skorlar, en_iyi_sonuc = motor.en_iyi_k_bul_kmeans(X, k_araligi=range(2, 5))

    assert en_iyi_k in [2, 3, 4]
    assert len(skorlar) == 3
    assert en_iyi_sonuc.kume_sayisi == en_iyi_k


def test_rapor_olusturma(ornek_gorseller, tmp_path):
    """Kümeleme görselleştirme raporunun PNG olarak kaydedildiğini test eder."""
    cikarici = GorselVektorCikarici(hedef_boyut=(32, 32))
    X = np.array([cikarici.cikar(img) for img in ornek_gorseller])

    motor = GorselKumelemeMotoru(random_state=42)
    sonuc = motor.k_means_kumele(X, n_clusters=3)

    hedef = tmp_path / "test_raporu.png"
    cikti = KumeGorsellestirici.kumeleme_raporu_olustur(
        X=X,
        gorseller=ornek_gorseller,
        kumeleme_sonucu=sonuc,
        k_skorlari={2: 0.5, 3: 0.8},
        hedef_dosya=hedef,
    )
    assert cikti.exists()
    assert cikti.stat().st_size > 0


def test_otomatik_epsilon_bul(ornek_gorseller):
    """Otomatik epsilon kestiriminin pozitif ve makul bir değer döndürdüğünü test eder."""
    cikarici = GorselVektorCikarici(hedef_boyut=(32, 32))
    X = np.array([cikarici.cikar(img) for img in ornek_gorseller])

    motor = GorselKumelemeMotoru()
    eps, k_mesafeleri = motor.otomatik_epsilon_bul(X, k=3, metric="cosine")

    assert isinstance(eps, float)
    assert eps > 0.0
    assert len(k_mesafeleri) == len(X)
    assert np.all(np.diff(k_mesafeleri) >= 0)  # Sıralı olmalı
