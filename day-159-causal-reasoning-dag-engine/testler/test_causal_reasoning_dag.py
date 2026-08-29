"""
GÜN 159: Causal Reasoning DAG & Do-Calculus Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.nedensel_dag_modeli import NedenselDAGModeli
from src.do_calculus_motoru import DoCalculusMotoru
from src.karsigelisci_akil_yurutucu import KarsigelisciAkilYurutucu
from src.gorsellestirici import NedensellikGorsellestirici


def test_dag_yapisi_dugumler_kenarlar():
    """DAG modelinin düğüm ve yönlü kenar yapısını test eder."""
    model = NedenselDAGModeli()

    assert "Z" in model.dugumler
    assert "X" in model.dugumler
    assert "Y" in model.dugumler
    assert ("Z", "X") in model.kenarlar
    assert ("X", "Y") in model.kenarlar


def test_gozlemsel_korelasyon_degerleri():
    """1. Basamak: Gözlemsel korelasyon olasılıklarını test eder."""
    model = NedenselDAGModeli()
    gozlem = model.gozlemsel_olasilik_hesapla()

    assert gozlem["p_y1_given_x1"] == 0.820
    assert gozlem["p_y1_given_x0"] == 0.480
    assert gozlem["gozlemsel_fark"] == 0.340


def test_do_calculus_mudahale_olasiliklari():
    """2. Basamak: Do-Calculus Backdoor müdahale değerlerini test eder."""
    model = NedenselDAGModeli()
    mudahale = DoCalculusMotoru.mudahale_etkisi_hesapla(model)

    assert mudahale["p_y1_do_x1"] == 0.700
    assert mudahale["p_y1_do_x0"] == 0.600


def test_ortalama_nedensel_etki_ate():
    """Gerçek Ortalama Nedensel Etkinin (ATE) +0.100 olduğunu test eder."""
    model = NedenselDAGModeli()
    mudahale = DoCalculusMotoru.mudahale_etkisi_hesapla(model)

    assert mudahale["ortalama_nedensel_etki_ate"] == 0.100


def test_simpson_paradoksu_tespiti():
    """Gözlemsel korelasyon ile gerçek nedensel etki arasındaki sapmayı test eder."""
    model = NedenselDAGModeli()
    gozlem = model.gozlemsel_olasilik_hesapla()
    mudahale = DoCalculusMotoru.mudahale_etkisi_hesapla(model)

    # Gözlem (0.340) > Gerçek ATE (0.100) -> Konfondör yanılsaması
    assert gozlem["gozlemsel_fark"] > mudahale["ortalama_nedensel_etki_ate"]


def test_karsigelisci_analiz_genc_birey():
    """3. Basamak: Genç bir birey için karşıgelişçi iyileşme ihtimalini test eder."""
    model = NedenselDAGModeli()
    karsigelisci = KarsigelisciAkilYurutucu.karsigelisci_analiz(
        model, birey_z=0, gerceklesen_x=1, gerceklesen_y=1, karsigelisci_x=0
    )

    assert karsigelisci["birey_yas_grubu"] == "Genç"
    assert karsigelisci["karsigelisci_iyilesme_olasiligi"] == 0.800


def test_zorunluluk_olasiligi_pn():
    """Zorunluluk olasılığının (PN) doğru hesaplandığını test eder."""
    model = NedenselDAGModeli()
    karsigelisci = KarsigelisciAkilYurutucu.karsigelisci_analiz(
        model, birey_z=0, gerceklesen_x=1, gerceklesen_y=1, karsigelisci_x=0
    )

    assert karsigelisci["zorunluluk_olasiligi_pn"] == 0.111


def test_gorsellestirici_pano_uretme():
    """6 panelli nedensellik teşhis panosunun PNG olarak kaydedildiğini test eder."""
    model = NedenselDAGModeli()
    gozlem = model.gozlemsel_olasilik_hesapla()
    mudahale = DoCalculusMotoru.mudahale_etkisi_hesapla(model)
    karsigelisci = KarsigelisciAkilYurutucu.karsigelisci_analiz(model)
    gorsellestirici = NedensellikGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_causal_pano.png")
        gorsellestirici.pano_olustur(gozlem, mudahale, karsigelisci, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
