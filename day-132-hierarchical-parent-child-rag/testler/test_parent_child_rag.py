"""
FAZ 7 GÜN 132: Hierarchical Parent-Child RAG Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.hiyerarsik_parcalayici import HiyerarsikParcalayici, EbeveynParca, CocukParca
from src.belge_deposu_ve_indeks import BelgeDeposu, VektorIndeksleyici
from src.parent_child_getirici import ParentChildRAGGetirici
from src.gorsellestirici import ParentChildGorsellestirici


def test_hiyerarsik_parcalayici_agac():
    """HiyerarsikParcalayici'nin ebeveyn ve çocuk parçaları üretip birbirine bağladığını test eder."""
    metin = "Bu bir uzun test belgesidir. " * 40
    parcalayici = HiyerarsikParcalayici(ebeveyn_boyutu=300, cocuk_boyutu=80)
    ebeveynler, cocuklar = parcalayici.hiyerarsi_olustur(metin)

    assert len(ebeveynler) >= 2
    assert len(cocuklar) > len(ebeveynler)
    assert cocuklar[0].parent_id == ebeveynler[0].parent_id
    assert len(ebeveynler[0].cocuk_idleri) >= 1


def test_belge_deposu_crud():
    """BelgeDeposu'nun ebeveyn parçaları Key-Value olarak başarıyla saklayıp getirdiğini test eder."""
    depo = BelgeDeposu()
    p1 = EbeveynParca(parent_id="P_01", metin="Ebeveyn metin 1", karakter_sayisi=16)
    p2 = EbeveynParca(parent_id="P_02", metin="Ebeveyn metin 2", karakter_sayisi=16)

    depo.toplu_ekle([p1, p2])
    assert depo.boyut() == 2

    getirilen = depo.getir("P_01")
    assert getirilen is not None
    assert getirilen.metin == "Ebeveyn metin 1"

    toplu = depo.toplu_getir(["P_01", "P_02", "P_99"])
    assert len(toplu) == 2


def test_vektor_indeksleyici_cocuk_arama():
    """VektorIndeksleyici'nin çocuk parçalar arasında en yakın benzerliği bulduğunu test eder."""
    motor = VektorIndeksleyici(vektor_boyutu=64)
    c1 = CocukParca(child_id="C_1", parent_id="P_1", metin="Veritabanı indeksleme ve SQL.", karakter_sayisi=27)
    c2 = CocukParca(child_id="C_2", parent_id="P_2", metin="Görüntü işleme ve CNN.", karakter_sayisi=23)

    motor.indeksle([c1, c2])
    sonuclar = motor.en_yakin_cocuklari_getir("SQL sorguları", top_k=1)

    assert len(sonuclar) == 1
    assert sonuclar[0][0].child_id == "C_1"
    assert sonuclar[0][1] > 0.0


def test_parent_child_getirici_belge_yukle():
    """ParentChildRAGGetirici sınıfının belgeyi yükleyip DocStore ve vektör indeksini doldurduğunu test eder."""
    getirici = ParentChildRAGGetirici(ebeveyn_boyutu=250, cocuk_boyutu=70)
    metin = "Bölüm A: Kuantum mekaniği. Bölüm B: Genel görelilik kuramı. Bölüm C: Termodinamik yasaları." * 5

    sayilar = getirici.belge_yukle(metin)
    assert sayilar["toplam_ebeveyn"] >= 1
    assert sayilar["toplam_cocuk"] >= sayilar["toplam_ebeveyn"]
    assert getirici.doc_store.boyut() == sayilar["toplam_ebeveyn"]


def test_parent_child_sorgula_ve_genislet_small_to_big():
    """Sorgulamanın çocuk parçadan başlayıp tam ebeveyn parçayı getirdiğini test eder."""
    getirici = ParentChildRAGGetirici(ebeveyn_boyutu=300, cocuk_boyutu=90)
    metin = (
        "Bölüm 1: Makine öğrenmesinde kayıp fonksiyonları optimize edilir. Gradyan inişi kullanılır. "
        "Bölüm 2: Biyolojide DNA çift sarmal yapıdadır. Genetik kod nükleotidlerle saklanır."
    )
    getirici.belge_yukle(metin)

    sonuc = getirici.sorgula_ve_genislet("DNA ve genetik yapı", cocuk_top_k=2)
    assert sonuc["eslesen_cocuk_sayisi"] >= 1
    assert sonuc["secilen_ebeveyn_sayisi"] >= 1
    assert "DNA" in sonuc["birlestirilmis_baglam"]
    assert "[KAYNAK: PARENT_" in sonuc["birlestirilmis_baglam"]


def test_parent_child_tekillestirme():
    """Aynı ebeveyne ait birden çok çocuk eşleştiğinde ebeveynin sadece 1 kez getirildiğini test eder."""
    getirici = ParentChildRAGGetirici(ebeveyn_boyutu=500, cocuk_boyutu=60)
    metin = "Python programlama dili çok yönlüdür. Yapay zekada standarttır. Pandas kütüphanesi çok popülerdir." * 3
    getirici.belge_yukle(metin)

    sonuc = getirici.sorgula_ve_genislet("Python kütüphanesi", cocuk_top_k=4)
    # Ebeveyn parçalar tekrarlanmamalıdır
    ebeveyn_idleri = [p["parent_id"] for p in sonuc["secilen_ebeveynler"]]
    assert len(ebeveyn_idleri) == len(set(ebeveyn_idleri))


def test_benchmark_karsilastir_metrikleri():
    """Parent-Child karşılaştırma metriklerinin eksiksiz olduğunu test eder."""
    getirici = ParentChildRAGGetirici()
    bench = getirici.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["hiyerarsik_parent_child"][0] > bench["duz_buyuk_parcalama"][0]


def test_parent_child_gorsellestirici_pano():
    """ParentChildGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    getirici = ParentChildRAGGetirici()
    metin = "Yapay zeka ve makine öğrenimi sistemleri. Veri bilimi ve Python." * 10
    getirici.belge_yukle(metin)
    sonuc = getirici.sorgula_ve_genislet("Yapay zeka")
    bench = getirici.benchmark_karsilastir()

    gorsellestirici = ParentChildGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_parent_child_pano.png")
        gorsellestirici.pano_olustur(sonuc, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
