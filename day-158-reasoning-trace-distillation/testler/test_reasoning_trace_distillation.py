"""
GÜN 158: Reasoning Trace Distillation Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.ogretmen_model_simulasyonu import OgretmenModelSimulasyonu
from src.iz_filtreleyici import DusunceIziFiltreleyici
from src.damitma_egitici import DamitmaEgitici
from src.gorsellestirici import DamitmaGorsellestirici


def test_ogretmen_iz_uretimi_mukemmel():
    """Öğretmen modelin eksiksiz düşünce izi ürettiğini test eder."""
    iz = OgretmenModelSimulasyonu.iz_uret("3x + 15 = 45", senaryo="mukemmel")

    assert iz["dogru_mu"] is True
    assert "<think>" in iz["ham_iz"]
    assert "</think>" in iz["ham_iz"]
    assert iz["refleksif_ifade_sayisi"] >= 1


def test_ogretmen_iz_uretimi_hatali():
    """Hatalı ve döngülü düşünce izlerinin doğru simüle edildiğini test eder."""
    iz = OgretmenModelSimulasyonu.iz_uret("3x + 15 = 45", senaryo="dongulu_hatali")

    assert iz["dogru_mu"] is False
    assert "Tekrar dene" in iz["ham_iz"]


def test_iz_filtreleyici_onaylama():
    """Kaliteli ve doğru düşünce izinin filtreden onay aldığını test eder."""
    iz = OgretmenModelSimulasyonu.iz_uret("3x + 15 = 45", senaryo="mukemmel")
    degerlendirme = DusunceIziFiltreleyici.izi_degerlendir(iz)

    assert degerlendirme["kabul_edildi_mi"] is True
    assert degerlendirme["kalite_skoru"] >= 0.85
    assert degerlendirme["red_nedeni"] is None


def test_iz_filtreleyici_red_hatali_sonuc():
    """Hatalı nihai sonuca sahip izlerin filtreden elendiğini test eder."""
    iz = OgretmenModelSimulasyonu.iz_uret("3x + 15 = 45", senaryo="hatali_sonuc")
    degerlendirme = DusunceIziFiltreleyici.izi_degerlendir(iz)

    assert degerlendirme["kabul_edildi_mi"] is False
    assert "yanlış" in degerlendirme["red_nedeni"].lower()


def test_iz_filtreleyici_red_dongu():
    """Kısırdöngüye giren izlerin filtreden elendiğini test eder."""
    iz = OgretmenModelSimulasyonu.iz_uret("3x + 15 = 45", senaryo="dongulu_hatali")
    degerlendirme = DusunceIziFiltreleyici.izi_degerlendir(iz)

    assert degerlendirme["kabul_edildi_mi"] is False
    assert "kısırdöngü" in degerlendirme["red_nedeni"].lower()


def test_damitma_egitici_loss_dususu():
    """SFT damıtma eğitiminde kayıp değerinin düştüğünü test eder."""
    sonuc = DamitmaEgitici.egitimi_simule_et()

    assert sonuc["kayiplar"][0] > sonuc["kayiplar"][-1]
    assert sonuc["final_sft_kayip"] < 0.50


def test_damitma_egitici_benchmark_sicramasi():
    """Damıtılmış öğrenci modelin ham modele göre büyük doğruluk sıçraması yaptığını test eder."""
    sonuc = DamitmaEgitici.egitimi_simule_et()

    assert sonuc["performans_kazanci_yuzde"] > 50.0
    assert sonuc["ogretmen_yakalama_orani"] > 85.0


def test_gorsellestirici_pano_uretme():
    """6 panelli damıtma teşhis panosunun PNG olarak kaydedildiğini test eder."""
    egitim = DamitmaEgitici.egitimi_simule_et()
    iz = OgretmenModelSimulasyonu.iz_uret("3x + 15 = 45", senaryo="mukemmel")
    gorsellestirici = DamitmaGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_distillation_pano.png")
        gorsellestirici.pano_olustur(egitim, iz, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
