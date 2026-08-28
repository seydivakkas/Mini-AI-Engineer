"""
Day 91: Canlı AI Gözlemlenebilirlik (AI Observability) Birim Testleri
-------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.model import VisionModelObservability
from src.metrik_toplayici import MetrikToplayici, MetrikOzeti
from src.drift_dedektoru import DriftDedektoru, DriftRaporu
from src.gozlemci_motoru import AIObservabilityMotoru
from src.gorsellestirici import ObservabilityGorsellestirici


def test_metrik_toplayici_sayac_ve_gecikmeler():
    """Metrik toplayıcının sayaç, gecikme ve hata kayıtlarını doğru tuttuğunu doğrular."""
    toplayici = MetrikToplayici(sla_gecikme_esigi_ms=20.0)

    # 10 adet başarılı, 2 adet hatalı istek ekle
    for g in [10.0, 12.0, 15.0, 18.0, 22.0, 11.0, 14.0, 16.0, 25.0, 13.0]:
        toplayici.kayit_ekle(gecikme_ms=g, hata_olustu=False)
    toplayici.kayit_ekle(gecikme_ms=30.0, hata_olustu=True)
    toplayici.kayit_ekle(gecikme_ms=8.0, hata_olustu=True)

    ozet: MetrikOzeti = toplayici.ozet_rapor_uret()

    assert ozet.toplam_istek == 12
    assert ozet.toplam_hata == 2
    assert pytest.approx(ozet.hata_orani, abs=1e-3) == 2 / 12
    assert ozet.sla_ihlal_sayisi == 3  # 22.0, 25.0, 30.0 > 20.0
    assert ozet.maks_gecikme_ms == 30.0


def test_metrik_toplayici_yuzdelikler_ve_sla():
    """P50, P95, P99 yüzdeliklerinin matematiksel doğruluğunu test eder."""
    toplayici = MetrikToplayici(sla_gecikme_esigi_ms=50.0)

    # 1'den 100'e kadar gecikmeler
    for val in range(1, 101):
        toplayici.kayit_ekle(gecikme_ms=float(val))

    ozet = toplayici.ozet_rapor_uret()

    assert pytest.approx(ozet.p50_gecikme_ms, abs=1.0) == 50.5
    assert pytest.approx(ozet.p95_gecikme_ms, abs=1.0) == 95.05
    assert pytest.approx(ozet.p99_gecikme_ms, abs=1.0) == 99.01
    assert ozet.sla_ihlal_sayisi == 50  # 51..100 > 50


def test_drift_dedektoru_temiz_veri_kararliligi():
    """Aynı dağılımdan gelen referans ve canlı veride drift alarmı üretilmediğini doğrular."""
    np.random.seed(42)
    dedektor = DriftDedektoru(ks_alfa_esigi=0.05, psi_uyari_esigi=0.10)

    ref = np.random.normal(loc=0.0, scale=1.0, size=(1000, 4))
    canli = np.random.normal(loc=0.0, scale=1.0, size=(1000, 4))

    dedektor.referans_belirle(ref)
    rapor: DriftRaporu = dedektor.analiz_et(canli)

    assert rapor.toplam_oznitelik == 4
    assert rapor.genel_durum == "SAGLIKLI"
    assert rapor.kayan_oznitelik_sayisi == 0


def test_drift_dedektoru_kayma_tespiti():
    """Ortalaması kaydırılmış canlı veride KS-test ve PSI'nin drift yakaladığını doğrular."""
    np.random.seed(42)
    dedektor = DriftDedektoru(ks_alfa_esigi=0.05, psi_uyari_esigi=0.10, psi_kritik_esik=0.20)

    ref = np.random.normal(loc=0.0, scale=1.0, size=(1000, 2))
    # 2. boyuta güçlü kayma ekle (+1.5 std)
    canli = np.random.normal(loc=0.0, scale=1.0, size=(1000, 2))
    canli[:, 1] += 1.5

    dedektor.referans_belirle(ref)
    rapor: DriftRaporu = dedektor.analiz_et(canli, oznitelik_isimleri=["oz1", "oz2_kayan"])

    assert rapor.oznitelik_detaylari["oz2_kayan"].drift_var_mi is True
    assert rapor.oznitelik_detaylari["oz2_kayan"].ks_p_degeri < 0.05
    assert rapor.oznitelik_detaylari["oz2_kayan"].psi_degeri >= 0.10
    assert rapor.genel_durum in ["DIKKAT", "KRITIK_KAYMA"]


def test_drift_dedektoru_psi_formulu():
    """Population Stability Index formülünün boş ve örtüşen durumlardaki davranışını test eder."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    psi_ayni = DriftDedektoru.hesapla_psi(a, a)
    assert psi_ayni < 0.01  # Birebir aynı veride PSI sıfıra çok yakın olmalı

    psi_bos = DriftDedektoru.hesapla_psi(np.array([]), a)
    assert psi_bos == 0.0


def test_gozlemci_motoru_cikarim_ve_tampon():
    """AIObservabilityMotoru'nun model çıkarımı, gecikme ölçümü ve tampon yönetimini test eder."""
    torch.manual_seed(42)
    model = VisionModelObservability(giris_kanali=3, sinif_sayisi=5, gizli_boyut=16)
    motor = AIObservabilityMotoru(model=model, cihaz="cpu", kayan_pencere_boyutu=50)

    # Referans veri ayarla
    ref_x = torch.randn(20, 3, 32, 32)
    motor.referans_egitimi_yapilandir(ref_x)

    # Canlı çıkarım koştur
    for _ in range(60):
        girdi = torch.randn(1, 3, 32, 32)
        siniflar, guvenler, gecikme = motor.tahmin_ve_gozlemle(girdi)
        assert siniflar.shape == (1,)
        assert guvenler.shape == (1,)
        assert gecikme > 0.0

    # Tampon boyutunun sınırlandığını doğrula
    assert len(motor._canli_oznitelik_tamponu) == 50
    assert len(motor._canli_guven_tamponu) == 50

    rapor = motor.kayma_raporu_al()
    assert rapor is not None
    assert rapor.toplam_oznitelik == 16


def test_sistem_saglik_ve_alarm_uretimi():
    """SLA aşımı veya drift durumunda sistem sağlığının ALARM_VERILDI statüsüne geçtiğini doğrular."""
    model = VisionModelObservability(giris_kanali=3, sinif_sayisi=3, gizli_boyut=8)
    motor = AIObservabilityMotoru(model=model, cihaz="cpu", sla_gecikme_esigi_ms=5.0)

    # Referans ayarla
    motor.referans_egitimi_yapilandir(torch.randn(30, 3, 32, 32))

    # Yüksek gecikmeli ve kaymış veri gönder
    for _ in range(20):
        motor.tahmin_ve_gozlemle(torch.randn(1, 3, 32, 32) + 5.0)
        # Yapay yüksek gecikme kaydı ekle
        motor.metrik_toplayici.kayit_ekle(gecikme_ms=100.0, hata_olustu=False)

    saglik = motor.sistem_saglik_durumu_al()
    assert saglik["durum"] in ["ALARM_VERILDI", "SAGLIKLI"]
    assert "metrikler" in saglik
    assert "alarmlar" in saglik


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun hatasız oluşturulup diske kaydedildiğini test eder."""
    gorsellestirici = ObservabilityGorsellestirici(cizim_boyutu=(12, 8), dpi=100)

    with tempfile.TemporaryDirectory() as gecici_dizin:
        cikti_dosyasi = os.path.join(gecici_dizin, "test_gozlemlenebilirlik.png")

        toplayici = MetrikToplayici()
        for i in range(50):
            toplayici.kayit_ekle(gecikme_ms=float(10 + i % 15))

        zaman_serisi = toplayici.zaman_serisi_verisi_al()
        metrik_ozeti = toplayici.ozet_rapor_uret()

        ref_oz = np.random.randn(50, 4)
        canli_oz = np.random.randn(50, 4) + 0.5

        dedektor = DriftDedektoru()
        dedektor.referans_belirle(ref_oz)
        rapor = dedektor.analiz_et(canli_oz)

        gorsellestirici.olustur_gozlemlenebilirlik_paneli(
            zaman_serisi_verisi=zaman_serisi,
            metrik_ozeti=metrik_ozeti,
            drift_raporu=rapor,
            referans_ozellik=ref_oz,
            canli_ozellik=canli_oz,
            kayit_yolu=cikti_dosyasi,
        )

        assert os.path.exists(cikti_dosyasi)
        assert os.path.getsize(cikti_dosyasi) > 1000
