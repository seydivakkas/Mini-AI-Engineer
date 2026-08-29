"""
FAZ 7 GÜN 122: Plan-and-Solve (PS / PS+) Prompting Testleri.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.planlayici_dag import AltGorev, GorevDAG
from src.araclar import AritmetikHesaplayici, VeriCikarici, MetinBirlestirici
from src.plan_and_solve_motoru import PlanAndSolveMotoru
from src.gorsellestirici import PlanAndSolveGorsellestirici


def test_alt_gorev_ve_dag_olusturma():
    """AltGorev nesnesinin ve GorevDAG yapısının doğru başlatıldığını test eder."""
    g1 = AltGorev(id="g1", tanim="İlk görev")
    dag = GorevDAG()
    dag.gorev_ekle(g1)

    assert "g1" in dag.gorevler
    assert dag.gorevler["g1"].durum == "bekliyor"


def test_dag_dongu_tespiti():
    """GorevDAG döngü (Cycle) tespit algoritmasının doğruluğunu test eder."""
    dag_gecerli = GorevDAG()
    dag_gecerli.gorev_ekle(AltGorev(id="a", tanim="A"))
    dag_gecerli.gorev_ekle(AltGorev(id="b", tanim="B", bagimliliklar=["a"]))
    assert dag_gecerli.dongu_var_mi() is False

    dag_dongulu = GorevDAG()
    dag_dongulu.gorev_ekle(AltGorev(id="x", tanim="X", bagimliliklar=["y"]))
    dag_dongulu.gorev_ekle(AltGorev(id="y", tanim="Y", bagimliliklar=["x"]))
    assert dag_dongulu.dongu_var_mi() is True


def test_topolojik_siralama_dogrulugu():
    """Topolojik sıralamanın bağımlılık sırasına riayet ettiğini test eder."""
    dag = GorevDAG()
    dag.gorev_ekle(AltGorev(id="adim_1", tanim="Gelir"))
    dag.gorev_ekle(AltGorev(id="adim_2", tanim="Maliyet"))
    dag.gorev_ekle(AltGorev(id="adim_3", tanim="Kar", bagimliliklar=["adim_1", "adim_2"]))

    sirali = dag.topolojik_sirala()
    id_listesi = [g.id for g in sirali]

    assert id_listesi.index("adim_1") < id_listesi.index("adim_3")
    assert id_listesi.index("adim_2") < id_listesi.index("adim_3")


def test_aritmetik_hesaplayici_degisken_ikamesi():
    """AritmetikHesaplayici sınıfının değişken ikamesi ve hesaplama doğruluğunu test eder."""
    hesaplayici = AritmetikHesaplayici()
    degiskenler = {"gelir": 1000.0, "maliyet": 400.0}
    sonuc = hesaplayici.hesapla("gelir - maliyet", degiskenler)
    assert sonuc == 600.0


def test_veri_cikarici():
    """VeriCikarici sınıfının metindeki sayıları doğru tespit ettiğini test eder."""
    cikarici = VeriCikarici()
    metin = "Ürün fiyatı 150 TL ve aylık 1200 adet satılıyor."
    sonuclar = cikarici.cikar(metin)

    assert "x_1" in sonuclar
    assert 150.0 in sonuclar.values()
    assert 1200.0 in sonuclar.values()


def test_metin_birlestirici_sentez():
    """MetinBirlestirici sınıfının sentez raporunu doğru ürettiğini test eder."""
    birlestirici = MetinBirlestirici()
    sonuclar = {"adim_1": 180000.0, "adim_2": 112000.0, "adim_3": 68000.0}
    sentez = birlestirici.sentezle("Kar Analizi", sonuclar)

    assert "Kar Analizi" in sentez
    assert "180000.0" in sentez
    assert "68000.0" in sentez


def test_plan_and_solve_motoru_cozum():
    """PlanAndSolveMotoru sınıfının 4 adımlı finansal problemi hatasız çözdüğünü test eder."""
    motor = PlanAndSolveMotoru()
    problem = "Bir ürünün birim fiyatı 150 TL, değişken maliyeti 60 TL, sabit maliyet 40000 TL ve 1200 adet satıldığında %20 vergi sonrası net kar nedir?"
    rapor = motor.coz(problem, mod="PS+")

    assert rapor["sirali_gorev_sayisi"] == 4
    # Gelir: 150 * 1200 = 180,000
    # Maliyet: 40000 + 60 * 1200 = 112,000
    # Brut Kar: 180,000 - 112,000 = 68,000
    # Net Kar (%20 vergi sonrası): 68,000 * 0.80 = 54,400
    assert rapor["durum_haritasi"]["adim_1_gelir"] == 180000.0
    assert rapor["durum_haritasi"]["adim_2_maliyet"] == 112000.0
    assert rapor["durum_haritasi"]["adim_3_brut_kar"] == 68000.0
    assert rapor["durum_haritasi"]["adim_4_vergi_sonrasi_net_kar"] == 54400.0
    assert rapor["nihai_deger"] == 54400.0


def test_gorsellestirici_pano():
    """PlanAndSolveGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    motor = PlanAndSolveMotoru()
    rapor = motor.coz("Finansal net kar hesapla", mod="PS+")
    karsilastirma = motor.benchmark_karsilastir()

    gorsellestirici = PlanAndSolveGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_ps_pano.png")
        gorsellestirici.pano_olustur(rapor, karsilastirma, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
