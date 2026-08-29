"""
GÜN 141: System 1 vs System 2 LLM Mimarisi Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.sistem1_motoru import Sistem1Motoru
from src.sistem2_motoru import Sistem2Motoru
from src.bilissel_karsilastirici import BilisselKarsilastirici
from src.gorsellestirici import System1VsSystem2Gorsellestirici


def test_sistem1_motoru_hizli_yanit():
    """Sistem1Motoru'nun düşünme tokeni harcamadan hızlı refleksif yanıt ürettiğini test eder."""
    motor = Sistem1Motoru()
    res = motor.yanitla("genel_soru", "Türkiye'nin başkenti neresidir?")

    assert res["sistem"] == "System 1 (Hızlı / Sezgisel)"
    assert res["dusunme_token_sayisi"] == 0
    assert len(res["dusunme_izleri"]) == 0
    assert res["gecikme_ms"] < 50.0


def test_sistem1_motoru_bilissel_tuzak():
    """Sistem1Motoru'nun Sopa ve Top probleminde sezgisel 10 cent tuzağına düştüğünü test eder."""
    motor = Sistem1Motoru()
    res = motor.yanitla("sopave_top", "Sopa ve top toplam $1.10...")

    assert "10 cent" in res["yanit"]


def test_sistem2_motoru_dusunme_izleri():
    """Sistem2Motoru'nun ara akıl yürütme adımlarını ürettiğini test eder."""
    motor = Sistem2Motoru(varsayilan_dusunme_butcesi=4)
    res = motor.yanitla("nilufer_golu", "Nilüfer gölü sorusu...", dusunme_butcesi=4)

    assert res["sistem"] == "System 2 (Yavaş / Akıl Yürüten)"
    assert len(res["dusunme_izleri"]) == 4
    assert res["dusunme_token_sayisi"] > 0
    assert len(res["adim_guven_egrisi"]) == 4


def test_sistem2_motoru_dogru_cozum():
    """Sistem2Motoru'nun adım adım mantıkla doğru matematiksel yanıta (5 cent) ulaştığını test eder."""
    motor = Sistem2Motoru()
    res = motor.yanitla("sopave_top", "Sopa ve top sorusu...")

    assert "5 cent" in res["yanit"]
    assert res["guven_skoru"] > 0.90


def test_sistem2_dusunme_butcesi_olcekleme():
    """Düşünme bütçesi parametresinin işletilen adım sayısını dinamik kısıtladığını test eder."""
    motor = Sistem2Motoru()
    res_2 = motor.yanitla("bes_makine", "5 makine...", dusunme_butcesi=2)
    res_4 = motor.yanitla("bes_makine", "5 makine...", dusunme_butcesi=4)

    assert len(res_2["dusunme_izleri"]) == 2
    assert len(res_4["dusunme_izleri"]) == 4
    assert res_4["dusunme_token_sayisi"] > res_2["dusunme_token_sayisi"]


def test_bilissel_karsilastirici_crt_benchmark():
    """BilisselKarsilastirici'nin CRT veri kümesinde System 1 vs System 2 ayrımını test eder."""
    karsilastirici = BilisselKarsilastirici()
    sonuc = karsilastirici.karsilastir()

    assert sonuc["toplam_soru"] == 3
    assert sonuc["sistem1"]["dogruluk_orani"] == 0.0  # Bilişsel tuzakların tamamına düştü
    assert sonuc["sistem2"]["dogruluk_orani"] == 100.0  # Mantıksal adımlarla tamamı çözüldü


def test_test_time_compute_olceklemesi():
    """Düşünme bütçesi arttıkça doğruluk oranının arttığını test eder."""
    karsilastirici = BilisselKarsilastirici()
    olcek = karsilastirici.test_time_compute_olceklemesi()

    assert len(olcek["butceler"]) == 4
    assert olcek["dogruluklar"][-1] == 100.0
    assert olcek["dusunme_tokenleri"][-1] > olcek["dusunme_tokenleri"][0]


def test_system1_vs_system2_gorsellestirici_pano():
    """6 panelli teşhis panosunun dosyaya başarıyla kaydedildiğini test eder."""
    karsilastirici = BilisselKarsilastirici()
    karsilastirma = karsilastirici.karsilastir()
    compute = karsilastirici.test_time_compute_olceklemesi()
    motor2 = Sistem2Motoru()
    ornek = motor2.yanitla("sopave_top", "Sopa top...")

    gorsellestirici = System1VsSystem2Gorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_system12_pano.png")
        gorsellestirici.pano_olustur(karsilastirma, compute, ornek, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
