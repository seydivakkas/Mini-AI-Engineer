"""
FAZ 7 GÜN 121: ReAct Ajanı ve Scratchpad Testleri.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.araclar import (
    HesapMakinasi,
    AramaMotoru,
    PythonCalistirici,
    AracKayitDefteri,
)
from src.react_ayristirici import ReActAyristirici
from src.scratchpad_bellek import ScratchpadBellek, AdimKaydi
from src.react_ajan import ReActAjani
from src.gorsellestirici import ReActGorsellestirici


def test_hesap_makinasi_guvenli_islem():
    """HesapMakinasi aracının aritmetik işlemleri doğru ve güvenli hesapladığını test eder."""
    hesap = HesapMakinasi()
    cikti = hesap.calistir("25 * 4 + 18")
    assert "118" in cikti

    hata_cikti = hesap.calistir("import os")
    assert "Hesaplama Hatası" in hata_cikti


def test_arama_motoru_bilgi_getirme():
    """AramaMotoru aracının bilgi tabanından ilgili kayıtları getirdiğini test eder."""
    arama = AramaMotoru()
    cikti = arama.calistir("türkiye başkenti")
    assert "Ankara" in cikti

    bos_cikti = arama.calistir("bilinmeyen garip sorgu 12345")
    assert "bulunamadı" in bos_cikti


def test_python_calistirici_guvenlik():
    """PythonCalistirici aracının güvenli kodları çalıştırıp tehlikeli çağrıları engellediğini test eder."""
    py = PythonCalistirici()
    cikti = py.calistir("sum([1, 2, 3, 4])")
    assert "10" in cikti

    engelli = py.calistir("import os; os.listdir('.')")
    assert "Güvenlik Uyarısı" in engelli


def test_arac_kayit_defteri_yonetimi():
    """AracKayitDefteri sınıfının araçları doğru kaydettiğini ve çalıştırdığını test eder."""
    kayit = AracKayitDefteri()
    kayit.arac_ekle(HesapMakinasi())
    kayit.arac_ekle(AramaMotoru())

    assert "HesapMakinasi" in kayit.araclar
    assert "AramaMotoru" in kayit.araclar

    sonuc = kayit.arac_calistir("HesapMakinasi", "10 / 2")
    assert "5.0" in sonuc

    hata = kayit.arac_calistir("OlmayanArac", "test")
    assert "Hata" in hata


def test_react_ayristirici_action_ve_thought():
    """ReActAyristirici sınıfının Thought ve Action[Input] formatını başarıyla ayrıştırdığını test eder."""
    metin = "Thought: Başkenti bulmalıyım.\nAction: AramaMotoru[türkiye başkenti]"
    sonuc = ReActAyristirici.ayristir(metin)

    assert sonuc["tip"] == "action"
    assert sonuc["dusunce"] == "Başkenti bulmalıyım."
    assert sonuc["arac_adi"] == "AramaMotoru"
    assert sonuc["arac_girdisi"] == "türkiye başkenti"


def test_react_ayristirici_final_answer():
    """ReActAyristirici sınıfının Final Answer formatını başarıyla ayrıştırdığını test eder."""
    metin = "Thought: Tüm bilgiler toplandı.\nFinal Answer: Türkiye'nin başkenti Ankara'dır."
    sonuc = ReActAyristirici.ayristir(metin)

    assert sonuc["tip"] == "final_answer"
    assert sonuc["dusunce"] == "Tüm bilgiler toplandı."
    assert "Ankara" in sonuc["nihai_yanit"]


def test_scratchpad_bellek_ekleme_ve_kaydirma():
    """ScratchpadBellek sınıfının adımları kaydettiğini ve sliding window uyguladığını test eder."""
    bellek = ScratchpadBellek(maksimum_adim=2)
    bellek.adim_ekle(AdimKaydi(adim_no=1, dusunce="D1", arac_adi="A1", arac_girdisi="G1", gozlem="O1"))
    bellek.adim_ekle(AdimKaydi(adim_no=2, dusunce="D2", arac_adi="A2", arac_girdisi="G2", gozlem="O2"))
    assert bellek.toplam_adim_sayisi() == 2

    bellek.adim_ekle(AdimKaydi(adim_no=3, dusunce="D3", nihai_yanit="Sonuc"))
    assert bellek.toplam_adim_sayisi() == 2
    assert bellek.son_adim().adim_no == 3


def test_react_ajan_gorev_calistirma_ve_karsilastirma():
    """ReActAjani sınıfının çok adımlı görevi başarıyla çözdüğünü ve mimari kıyas ürettiğini test eder."""
    kayit = AracKayitDefteri()
    kayit.arac_ekle(HesapMakinasi())
    kayit.arac_ekle(AramaMotoru())

    ajan = ReActAjani(arac_kayit=kayit, maksimum_iterasyon=5)
    sonuc = ajan.calistir("türkiye başkenti ve nüfus bilgisi")

    assert sonuc["basarili"] is True
    assert sonuc["toplam_adim"] >= 2
    assert "Ankara" in sonuc["nihai_yanit"]

    karsilastirma = ajan.mimari_karsilastir()
    assert "ReAct (Düşünce + Eylem)" in karsilastirma["modeller"]
    assert karsilastirma["dogruluk_orani"][2] > karsilastirma["dogruluk_orani"][0]
