"""
Day 42: Üretim Girdi Tensörleri Doğrulama ve Anomali Denetçisi Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.sema import TensorSemasi
from src.denetleyici import AIBatchDenetleyici
from src.temizleyici import BatchTemizleyici
from src.gorsellestirici import TensorDenetimGorsellestirici


@pytest.fixture
def standart_sema():
    return TensorSemasi(
        model_adi="TestModel",
        beklenen_sekil=(-1, 3, 32, 32),
        kanal_sirasi="NCHW",
        gecerli_tipler=[np.float32],
        deger_araligi=(-3.0, 3.0),
        max_batch=16,
        sureklilik_sarti=True,
        max_bellek_mb=10.0
    )


def test_gecerli_batch_denetimi(standart_sema):
    """Kusursuz girdi tensörünün GECERLI olarak onaylandığını test eder."""
    denetleyici = AIBatchDenetleyici(standart_sema)
    tensor = np.zeros((4, 3, 32, 32), dtype=np.float32)
    rapor = denetleyici.denetle(tensor)

    assert rapor["gecerli"] is True if "gecerli" in rapor else rapor["guvenli_gecis"] is True
    assert rapor["karar"] in ["GECERLI", "DUZELTILEBILIR_UYARI"]
    assert rapor["istatistikler"]["nan_sayisi"] == 0


def test_nan_tespiti_ve_reddi(standart_sema):
    """NaN içeren tensörün anında KRITIK_RED aldığını test eder."""
    denetleyici = AIBatchDenetleyici(standart_sema)
    tensor = np.zeros((2, 3, 32, 32), dtype=np.float32)
    tensor[0, 0, 5, 5] = np.nan
    rapor = denetleyici.denetle(tensor)

    assert rapor["karar"] == "KRITIK_RED"
    assert rapor["guvenli_gecis"] is False
    assert rapor["istatistikler"]["nan_sayisi"] == 1


def test_inf_tespiti_ve_reddi(standart_sema):
    """Sonsuz (Inf) değer içeren tensörün reddedildiğini test eder."""
    denetleyici = AIBatchDenetleyici(standart_sema)
    tensor = np.zeros((2, 3, 32, 32), dtype=np.float32)
    tensor[0, 1, 10, 10] = np.inf
    rapor = denetleyici.denetle(tensor)

    assert rapor["karar"] == "KRITIK_RED"
    assert rapor["guvenli_gecis"] is False
    assert rapor["istatistikler"]["inf_sayisi"] == 1


def test_batch_boyutu_asimi(standart_sema):
    """Maksimum batch sınırını aşan tensörün yakalandığını test eder."""
    denetleyici = AIBatchDenetleyici(standart_sema)
    buyuk_tensor = np.zeros((32, 3, 32, 32), dtype=np.float32)  # max_batch=16
    rapor = denetleyici.denetle(buyuk_tensor)

    assert rapor["karar"] == "KRITIK_RED"
    assert any("Batch boyutu limit dışı" in ih["mesaj"] for ih in rapor["ihlaller"])


def test_nhwc_tespiti_ve_otomatik_temizleme(standart_sema):
    """NHWC kanal düzeninin tespit edilip NCHW'ye dönüştürüldüğünü test eder."""
    denetleyici = AIBatchDenetleyici(standart_sema)
    temizleyici = BatchTemizleyici(standart_sema)

    nhwc_tensor = np.zeros((4, 32, 32, 3), dtype=np.float32)
    rapor = denetleyici.denetle(nhwc_tensor)

    assert rapor["nhwc_tespit_edildi"] is True
    temiz_tensor, temiz_rapor = temizleyici.temizle_ve_uyarla(nhwc_tensor, rapor)
    assert temiz_tensor.shape == (4, 3, 32, 32)
    assert "KANAL_DUZENI_TRANSPOSE_NHWC_TO_NCHW" in temiz_rapor["yapilan_islemler"]


def test_deger_kirpma_ve_dtype_donusturme(standart_sema):
    """Aralık dışı float64 değerlerin kırpılıp float32'ye uyarlandığını test eder."""
    denetleyici = AIBatchDenetleyici(standart_sema)
    temizleyici = BatchTemizleyici(standart_sema)

    outlier_tensor = np.array([[[[10.5]]]], dtype=np.float64)  # Aralık [-3, 3]
    outlier_tensor = np.zeros((1, 3, 32, 32), dtype=np.float64)
    outlier_tensor[0, 0, 0, 0] = 8.5

    rapor = denetleyici.denetle(outlier_tensor)
    assert rapor["istatistikler"]["aralik_disi_piksel"] == 1

    temiz_tensor, _ = temizleyici.temizle_ve_uyarla(outlier_tensor, rapor)
    assert temiz_tensor.dtype == np.float32
    assert float(np.max(temiz_tensor)) <= 3.0


def test_gorsellestirici_png_uretimi(standart_sema, tmp_path):
    """6 panelli teşhis panosunun başarıyla PNG ürettiğini test eder."""
    denetleyici = AIBatchDenetleyici(standart_sema)
    tensor = np.zeros((2, 3, 32, 32), dtype=np.float32)
    rapor = denetleyici.denetle(tensor)

    cikis_yolu = str(tmp_path / "test_denetim_paneli.png")
    yol = TensorDenetimGorsellestirici.panel_ciz(tensor, rapor, hedef_path=cikis_yolu)
    assert os.path.exists(yol)
