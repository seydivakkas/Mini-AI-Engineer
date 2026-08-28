"""Day 20 Birim Testleri: TensorFlow/Keras ile CNN Görsel Sınıflandırma."""

import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import numpy as np
import pytest

from src.model_mimari import build_cnn_model
from src.veri_hazirlayici import VeriHazirlayici
from src.egitici import ModelEgitici, EgitimSonucu
from src.gorsellestirici import CNNGorsellestirici


def test_model_mimarisi_katmanlar_ve_cikti():
    """CNN modelinin katmanlarının ve çıktı tensör şeklinin doğruluğunu test eder."""
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=3)

    assert model is not None
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 3)
    assert model.count_params() > 0


def test_model_gecersiz_parametreler():
    """Geçersiz girdi şekli veya sınıf sayısı verildiğinde ValueError fırlatıldığını doğrular."""
    with pytest.raises(ValueError):
        build_cnn_model(input_shape=(32, 32), num_classes=3)

    with pytest.raises(ValueError):
        build_cnn_model(input_shape=(32, 32, 3), num_classes=1)


def test_veri_hazirlayici_sentetik_uretim():
    """Veri hazırlayıcının doğru boyut ve aralıkta görsel ürettiğini test eder."""
    hazirlayici = VeriHazirlayici(hedef_boyut=(32, 32), random_state=42)
    X, y, siniflar = hazirlayici.sentetik_veri_seti_uret(sinif_basina_ornek=4)

    assert len(X) == 16
    assert X.shape == (16, 32, 32, 3)
    assert X.dtype == np.float32
    assert 0.0 <= X.min() <= X.max() <= 1.0
    assert len(siniflar) == 4


def test_veri_hazirlayici_bolumleme():
    """Veri bölümlemenin train/val/test boyutlarını doğru ayırdığını doğrular."""
    hazirlayici = VeriHazirlayici(hedef_boyut=(32, 32), random_state=42)
    X = np.zeros((40, 32, 32, 3), dtype=np.float32)
    y = np.repeat([0, 1, 2, 3], 10)

    X_train, y_train, X_val, y_val, X_test, y_test = hazirlayici.veri_bol(
        X, y, val_orani=0.15, test_orani=0.15
    )

    assert len(X_train) + len(X_val) + len(X_test) == 40
    assert len(X_train) == 28
    assert len(X_val) == 6
    assert len(X_test) == 6


def test_model_egitimi_ve_degerlendirme():
    """Modelin mini bir veri üzerinde hızlıca eğitilip değerlendirildiğini test eder."""
    hazirlayici = VeriHazirlayici(hedef_boyut=(32, 32), random_state=42)
    X, y, _ = hazirlayici.sentetik_veri_seti_uret(sinif_basina_ornek=6)
    X_train, y_train, X_val, y_val, X_test, y_test = hazirlayici.veri_bol(
        X, y, val_orani=0.20, test_orani=0.20
    )

    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=4)
    egitici = ModelEgitici(model)

    tarihce = egitici.egit(X_train, y_train, X_val, y_val, epochs=2, batch_size=8)
    assert "loss" in tarihce.history
    assert "val_loss" in tarihce.history

    sonuc = egitici.degerlendir(X_test, y_test, tarihce, egitim_suresi_sn=1.0)
    assert isinstance(sonuc, EgitimSonucu)
    assert 0.0 <= sonuc.test_dogruluk <= 1.0
    assert 0.0 <= sonuc.f1_macro <= 1.0


def test_gorsellestirici_rapor_cizimi(tmp_path):
    """Teşhis raporunun PNG formatında üretilip kaydedildiğini test eder."""
    hazirlayici = VeriHazirlayici(hedef_boyut=(32, 32), random_state=42)
    X, y, siniflar = hazirlayici.sentetik_veri_seti_uret(sinif_basina_ornek=4)
    X_train, y_train, X_val, y_val, X_test, y_test = hazirlayici.veri_bol(X, y)

    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=4)
    egitici = ModelEgitici(model)
    tarihce = egitici.egit(X_train, y_train, X_val, y_val, epochs=1, batch_size=8)
    sonuc = egitici.degerlendir(X_test, y_test, tarihce, egitim_suresi_sn=0.5)

    hedef = tmp_path / "test_cnn_rapor.png"
    cikti = CNNGorsellestirici.egitim_raporu_ciz(sonuc, siniflar, X_test, hedef_dosya=hedef)

    assert cikti.exists()
    assert cikti.stat().st_size > 0


def test_ara_katman_aktivasyon_cikarici(tmp_path):
    """Ara katman aktivasyon haritalarının doğru şekil ve grid formatında çıkarıldığını test eder."""
    from src.aktivasyon_cikarici import AraKatmanAktivasyonCikarici

    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=4)
    cikarici = AraKatmanAktivasyonCikarici(model)

    test_gorsel = np.random.rand(32, 32, 3).astype(np.float32)
    aktivasyon = cikarici.aktivasyon_haritasi_cikar(test_gorsel, katman_adi="conv2d_blok1")

    assert aktivasyon.shape == (1, 32, 32, 32)
    fig = cikarici.aktivasyon_grid_ciz(aktivasyon, maks_filtre=8)
    assert fig is not None
