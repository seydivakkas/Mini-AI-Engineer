"""
GÜN 168: Gerçek Zamanlı Video Akışı Analizi Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.kayan_bellek_kuyrugu import KayanBellekKuyrugu
from src.olay_tetikleyici_dedektor import OlayTetikleyiciDedektor
from src.streaming_vlm_motoru import StreamingVLMMotoru
from src.gorsellestirici import StreamingGorsellestirici


def test_kayan_bellek_kuyrugu_kapasite_siniri():
    """Kuyruğun maksimum kapasiteyi (16) aşmadığını ve eski kareleri FIFO mantığıyla attığını test eder."""
    kuyruk = KayanBellekKuyrugu(maks_kapasite=4)
    for i in range(10):
        kuyruk.kare_ekle(torch.tensor([i]), zaman_damgasi=float(i))

    assert len(kuyruk) == 4
    aktif = kuyruk.aktif_pencereyi_getir()
    assert aktif[0]["zaman_damgasi"] == 6.0
    assert aktif[-1]["zaman_damgasi"] == 9.0


def test_kayan_bellek_tensor_yigini():
    """Tampondaki tensorların doğru boyutlu tek bir batch'e dönüştüğünü test eder."""
    kuyruk = KayanBellekKuyrugu(maks_kapasite=4)
    for i in range(4):
        kuyruk.kare_ekle(torch.randn(16, 32), zaman_damgasi=float(i))

    yigin = kuyruk.tensor_yigini_olustur()
    assert yigin.shape == (1, 4, 16, 32)


def test_olay_tetikleyici_esik_alti():
    """Durağan veya hafif gürültülü sahnelerde tetikleme yapılmadığını test eder."""
    dedektor = OlayTetikleyiciDedektor(degisim_esigi=0.40)
    v1 = torch.ones(32)
    v2 = torch.ones(32) + torch.randn(32) * 0.05
    tetik1, _ = dedektor.olay_tetiklendi_mi(v1)
    tetik2, skor2 = dedektor.olay_tetiklendi_mi(v2)

    assert tetik1 is False
    assert tetik2 is False
    assert skor2 < 0.40


def test_olay_tetikleyici_esik_ustu_anomali():
    """Ani görsel değişimde (anomali) olayın tetiklendiğini test eder."""
    dedektor = OlayTetikleyiciDedektor(degisim_esigi=0.30)
    v1 = torch.ones(32)
    v2 = -torch.ones(32)  # Zıt yönde vektör (mesafe ~ 2.0)
    dedektor.olay_tetiklendi_mi(v1)
    tetik, skor = dedektor.olay_tetiklendi_mi(v2)

    assert tetik is True
    assert skor >= 0.30


def test_streaming_vlm_canli_akis_simulasyonu():
    """Streaming motorunun 30 saniyelik akışı hatasız işlediğini test eder."""
    rapor = StreamingVLMMotoru.canli_akis_simulasyonunu_calistir()

    assert rapor["toplam_islenen_saniye"] == 30
    assert rapor["toplam_tetiklenen_olay"] == 2
    assert len(rapor["olay_gunlugu"]) == 2
    assert rapor["dogruluk_yuzdesi"] == 100.0


def test_streaming_vlm_kare_isleme_formati():
    """Tek kare işleme fonksiyonunun beklenen anahtarları döndürdüğünü test eder."""
    motor = StreamingVLMMotoru(pencere_kapasitesi=8, tetikleme_esigi=0.35)
    sonuc = motor.kare_isle(torch.randn(10), 1.0, torch.randn(32), "test olayı")

    assert "zaman_damgasi" in sonuc
    assert "tetiklendi" in sonuc
    assert "anomali_skoru" in sonuc
    assert "alarm_seviyesi" in sonuc
    assert "vlm_aciklamasi" in sonuc


def test_kayan_bellek_bos_hata():
    """Boş kuyrukta tensor yığını istendiğinde ValueError fırlatıldığını test eder."""
    kuyruk = KayanBellekKuyrugu(maks_kapasite=4)
    with pytest.raises(ValueError):
        kuyruk.tensor_yigini_olustur()


def test_gorsellestirici_pano_uretme():
    """6 panelli Streaming VLM teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = StreamingVLMMotoru.canli_akis_simulasyonunu_calistir()
    gorsellestirici = StreamingGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_streaming_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
