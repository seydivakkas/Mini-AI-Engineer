"""
GÜN 147: Test-Time Compute Scaling Yasaları Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.scaling_yasa_modeli import TestTimeScalingModeli
from src.test_time_hesaplayici import TestTimeHesaplayici
from src.pareto_sinir_analizcisi import ParetoSinirAnalizcisi
from src.gorsellestirici import TestTimeScalingGorsellestirici


def test_scaling_modeli_hata_hesaplama():
    """TestTimeScalingModeli'nin bütçe arttıkça hata oranını azalttığını test eder."""
    model = TestTimeScalingModeli(alfa=0.65, beta=0.42, gama=0.05)
    hata_1 = model.hata_hesapla(1)
    hata_16 = model.hata_hesapla(16)
    hata_64 = model.hata_hesapla(64)

    assert hata_1 > hata_16 > hata_64
    assert hata_64 >= 0.05


def test_scaling_modeli_dogruluk_artisi():
    """TestTimeScalingModeli'nin bütçe arttıkça doğruluğu monoton artırdığını test eder."""
    model = TestTimeScalingModeli()
    acc_1 = model.dogruluk_hesapla(1)
    acc_100 = model.dogruluk_hesapla(100)

    assert acc_100 > acc_1
    assert 0.0 <= acc_100 <= 1.0


def test_scaling_modeli_butce_taramasi():
    """Bütçe taramasının doğru sayıda ve yapıda sonuç döndürdüğünü test eder."""
    model = TestTimeScalingModeli()
    butceler = [1, 4, 16, 64, 256]
    sonuclar = model.butce_taramasi(butceler)

    assert len(sonuclar) == 5
    assert sonuclar[0]["butce_n"] == 1
    assert sonuclar[-1]["butce_n"] == 256
    assert sonuclar[-1]["dogruluk_orani"] > sonuclar[0]["dogruluk_orani"]


def test_test_time_hesaplayici_butce_dagitimi():
    """TestTimeHesaplayici'nin token bütçesini 3 farklı arama stratejisine paylaştırdığını test eder."""
    analiz = TestTimeHesaplayici.butce_dagitimi_analiz_et(toplam_token_butcesi=4096, adim_basi_token=128)

    assert analiz["toplam_adim_kapasitesi"] == 32
    assert "paralel_ornekleme" in analiz
    assert "derin_sirali_arama" in analiz
    assert "dengeli_agac_aramasi" in analiz


def test_test_time_hesaplayici_agac_ustunlugu():
    """Dengeli ağaç aramasının sığ paralel örneklemeden daha yüksek doğruluk ürettiğini test eder."""
    analiz = TestTimeHesaplayici.butce_dagitimi_analiz_et(toplam_token_butcesi=4096, adim_basi_token=128)
    acc_agac = analiz["dengeli_agac_aramasi"]["tahmini_dogruluk"]
    acc_paralel = analiz["paralel_ornekleme"]["tahmini_dogruluk"]

    assert acc_agac >= acc_paralel


def test_pareto_sinir_analizcisi_optimal_noktalar():
    """ParetoSinirAnalizcisi'nin 8B ve 70B senaryolarını ve Pareto optimalliğini test eder."""
    senaryolar = ParetoSinirAnalizcisi.pareto_karsilastirmasi()

    assert len(senaryolar) == 8
    assert any(s["pareto_optimal_mi"] for s in senaryolar)


def test_pareto_sinir_8b_ile_70b_gecisi():
    """8B + 16x Test-Time Compute'un 70B (1x Base) model doğruluğunu geçtiğini test eder."""
    senaryolar = ParetoSinirAnalizcisi.pareto_karsilastirmasi()
    s_8b_16x = next(s for s in senaryolar if s["model"] == "8B" and s["test_compute"] == 16)
    s_70b_1x = next(s for s in senaryolar if s["model"] == "70B" and s["test_compute"] == 1)

    assert s_8b_16x["dogruluk"] > s_70b_1x["dogruluk"]
    assert s_8b_16x["bellek_gb"] < s_70b_1x["bellek_gb"]


def test_gorsellestirici_pano_uretme():
    """6 panelli Test-Time Compute Scaling teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    model = TestTimeScalingModeli()
    scaling_verileri = model.butce_taramasi([1, 4, 16, 64, 256])
    pareto_verileri = ParetoSinirAnalizcisi.pareto_karsilastirmasi()
    butce_analizi = TestTimeHesaplayici.butce_dagitimi_analiz_et(toplam_token_butcesi=4096)

    gorsellestirici = TestTimeScalingGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_scaling_pano.png")
        gorsellestirici.pano_olustur(scaling_verileri, pareto_verileri, butce_analizi, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
