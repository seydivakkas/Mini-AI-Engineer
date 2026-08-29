"""
FAZ 7 GÜN 130: Human-in-the-Loop (HITL) Kesinti ve Güvenlik Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.risk_ve_eylem_semasi import EylemSeviyesi, RiskSiniflandirici
from src.hitl_kesinti_motoru import HITLOrkestratoru
from src.gorsellestirici import HITLGorsellestirici


def test_risk_siniflandirici_dusuk_risk():
    """Güvenli eylemlerin DÜŞÜK risk seviyesi ve onaysız icra olarak sınıflandırıldığını test eder."""
    eylem = RiskSiniflandirici.eylemi_degerlendir("log_sorgula", {"limit": 50})
    assert eylem.seviye == EylemSeviyesi.DUSUK
    assert eylem.risk_skoru < 0.30
    assert eylem.onay_gerekli_mi is False


def test_risk_siniflandirici_kritik_risk():
    """Tehlikeli eylemlerin (tablo silme, yüksek para transferi) KRİTİK risk olarak işaretlendiğini test eder."""
    eylem1 = RiskSiniflandirici.eylemi_degerlendir("veritabani_tablo_sil", {"tablo": "musteriler"})
    assert eylem1.seviye == EylemSeviyesi.KRITIK
    assert eylem1.onay_gerekli_mi is True

    eylem2 = RiskSiniflandirici.eylemi_degerlendir("para_transferi", {"tutar": 75000.0})
    assert eylem2.seviye == EylemSeviyesi.KRITIK
    assert eylem2.onay_gerekli_mi is True


def test_hitl_orkestratoru_otomatik_icra():
    """Düşük riskli eylemlerin insan müdahalesine gerek kalmadan doğrudan tamamlandığını test eder."""
    ork = HITLOrkestratoru()
    ork.eylem_ekle("log_sorgula", {"modul": "auth"})
    ork.eylem_ekle("rapor_olustur", {"format": "pdf"})

    sonuc = ork.adim_adim_calistir()
    assert sonuc["durum"] == "TUM_GOREVLER_TAMAMLANDI"
    assert sonuc["kesinti_var_mi"] is False
    assert len(ork.tamamlanan_eylemler) == 2


def test_hitl_orkestratoru_kesinti_ve_onay():
    """Yüksek riskli eylemde çizgenin durakladığını (Interrupt) ve 'ONAYLA' sonrası çalıştığını test eder."""
    ork = HITLOrkestratoru()
    ork.eylem_ekle("toplu_eposta_gonder", {"adet": 5000})

    # 1. Faz: Kesinti Bekle
    durum1 = ork.adim_adim_calistir()
    assert durum1["durum"] == "BEKLIYOR_INSAN_ONAYI"
    assert durum1["kesinti_var_mi"] is True
    assert durum1["kesinti_eylemi"].eylem_adi == "toplu_eposta_gonder"

    # 2. Faz: İnsan Onayı Ver
    durum2 = ork.insan_karari_isle("ONAYLA")
    assert durum2["durum"] == "TUM_GOREVLER_TAMAMLANDI"
    assert len(ork.tamamlanan_eylemler) == 1
    assert ork.tamamlanan_eylemler[0].insan_karari == "ONAYLANDI"


def test_hitl_orkestratoru_reddet_ve_guvenli_alternatif():
    """Kritik eylem reddedildiğinde yıkıcı işlemin engellenip güvenli alternatif üretildiğini test eder."""
    ork = HITLOrkestratoru()
    ork.eylem_ekle("veritabani_tablo_sil", {"tablo": "odemeler"})

    durum1 = ork.adim_adim_calistir()
    assert durum1["kesinti_var_mi"] is True

    # Denetçi Reddediyor
    durum2 = ork.insan_karari_isle("REDDET", red_gerekcesi="Canlı veritabanı silinemez")
    assert durum2["durum"] == "TUM_GOREVLER_TAMAMLANDI"
    assert ork.tamamlanan_eylemler[0].insan_karari == "REDDEDILDI"
    # Güvenli alternatif eklendi mi?
    assert ork.tamamlanan_eylemler[1].eylem_adi == "rapor_olustur"


def test_hitl_orkestratoru_duzenle_ve_icra():
    """Denetçinin eylem parametrelerini güvenli hale getirip ('DUZENLE') onayladığını test eder."""
    ork = HITLOrkestratoru()
    ork.eylem_ekle("para_transferi", {"tutar": 95000.0, "alici": "HESAP_X"})

    ork.adim_adim_calistir()
    # Tutarı 8000 TL'ye düşürerek düzenle
    durum = ork.insan_karari_isle("DUZENLE", yeni_parametreler={"tutar": 8000.0, "alici": "HESAP_X"})

    assert durum["durum"] == "TUM_GOREVLER_TAMAMLANDI"
    tamamlanan = ork.tamamlanan_eylemler[0]
    assert tamamlanan.insan_karari == "DUZENLENDI"
    assert tamamlanan.parametreler["tutar"] == 8000.0


def test_benchmark_karsilastir_metrikleri():
    """HITL güvenlik karşılaştırma metriklerinin eksiksiz olduğunu test eder."""
    ork = HITLOrkestratoru()
    bench = ork.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["human_in_the_loop_ajan"][0] == 100.0


def test_hitl_gorsellestirici_pano():
    """HITLGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    ork = HITLOrkestratoru()
    ork.eylem_ekle("log_sorgula", {"limit": 10})
    ork.adim_adim_calistir()
    bench = ork.benchmark_karsilastir()

    gorsellestirici = HITLGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_hitl_pano.png")
        gorsellestirici.pano_olustur(ork.denetim_izi, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
