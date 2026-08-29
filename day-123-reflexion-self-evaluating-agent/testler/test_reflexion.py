"""
FAZ 7 GÜN 123: Reflexion Ajanı ve Sözel Öz-Eleştiri Testleri.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.degerlendirici import KodDegerlendirici, TestDurumu
from src.oz_elestiri_ureteci import OzElestiriUreteci
from src.hafiza_tamponu import ReflexionHafizaTamponu, DenemeKaydi
from src.reflexion_ajani import ReflexionAjani
from src.gorsellestirici import ReflexionGorsellestirici


def test_kod_degerlendirici_basarili_fonksiyon():
    """KodDegerlendirici sınıfının hatasız çalışan fonksiyona 1.0 tam ödül verdiğini test eder."""
    degerlendirici = KodDegerlendirici()
    kod = "def topla(a, b):\n    return a + b\n"
    testler = [
        TestDurumu(girdi=(2, 3), beklenen=5, aciklama="Pozitif sayılar"),
        TestDurumu(girdi=(-1, 1), beklenen=0, aciklama="Negatif sayı"),
    ]
    sonuc = degerlendirici.degerlendir(kod, "topla", testler)

    assert sonuc["basarili"] is True
    assert sonuc["odul"] == 1.0
    assert sonuc["gecen_test"] == 2


def test_kod_degerlendirici_hatali_fonksiyon():
    """KodDegerlendirici sınıfının mantıksal hatalı fonksiyonda kısmi ödül ve hata mesajı ürettiğini test eder."""
    degerlendirici = KodDegerlendirici()
    kod = "def carp(a, b):\n    return a + b  # HATA: Çarpma yerine toplama yapılmış\n"
    testler = [
        TestDurumu(girdi=(2, 2), beklenen=4, aciklama="2*2=4"),
        TestDurumu(girdi=(2, 3), beklenen=6, aciklama="2*3=6"),
    ]
    sonuc = degerlendirici.degerlendir(kod, "carp", testler)

    assert sonuc["basarili"] is False
    assert sonuc["odul"] == 0.5  # 2*2=4 geçer, 2*3=6 kalır
    assert "AssertionMismatch" in sonuc["hata_tipi"]


def test_kod_degerlendirici_syntax_ve_bulunamadi():
    """KodDegerlendirici sınıfının sözdizimi ve fonksiyon bulunamadı hatalarını güvenle yakaladığını test eder."""
    degerlendirici = KodDegerlendirici()
    sonuc_syntax = degerlendirici.degerlendir("def bozuk_kod(:\n pass", "f", [])
    assert sonuc_syntax["basarili"] is False
    assert "Syntax/CompileError" in sonuc_syntax["hata_tipi"]

    sonuc_yok = degerlendirici.degerlendir("def var_olan(): pass", "olmayan_fonk", [])
    assert sonuc_yok["basarili"] is False
    assert "FunctionNotFoundError" in sonuc_yok["hata_tipi"]


def test_oz_elestiri_ureteci_assertion_elestirisi():
    """OzElestiriUreteci sınıfının hata mesajına göre anlamlı sözel eleştiri ürettiğini test eder."""
    reflector = OzElestiriUreteci()
    degerlendirme = {
        "hata_tipi": "AssertionMismatch",
        "hata_mesaji": "Test Başarısız: Girdi=[-2, -1], Beklenen=-1, Gerçekleşen=0",
    }
    elestiri = reflector.elestiri_uret(1, "def f(): pass", degerlendirme)

    assert "Öz-Eleştiri (Trial 1)" in elestiri
    assert "negatif" in elestiri.lower() or "döngü" in elestiri.lower()


def test_hafiza_tamponu_ekleme_ve_prompt_olusturma():
    """ReflexionHafizaTamponu sınıfının geçmiş denemeleri saklayıp prompt oluşturduğunu test eder."""
    hafiza = ReflexionHafizaTamponu(maksimum_hafiza=3)
    hafiza.deneme_ekle(DenemeKaydi(deneme_no=1, kod="kod1", odul=0.5, hata_mesaji="Hata 1", oz_elestiri="Ders 1"))

    prompt_metni = hafiza.prompt_gecmisi_olustur()
    assert "EPISODIC MEMORY" in prompt_metni
    assert "Deneme 1" in prompt_metni
    assert "Ders 1" in prompt_metni


def test_hafiza_tamponu_en_iyi_deneme():
    """ReflexionHafizaTamponu sınıfının en yüksek ödüllü denemeyi doğru bulduğunu test eder."""
    hafiza = ReflexionHafizaTamponu()
    hafiza.deneme_ekle(DenemeKaydi(deneme_no=1, kod="k1", odul=0.3, hata_mesaji="", oz_elestiri=""))
    hafiza.deneme_ekle(DenemeKaydi(deneme_no=2, kod="k2", odul=0.9, hata_mesaji="", oz_elestiri=""))

    en_iyi = hafiza.en_iyi_deneme()
    assert en_iyi.deneme_no == 2
    assert en_iyi.odul == 0.9


def test_reflexion_ajani_iteratif_cozum():
    """ReflexionAjani sınıfının Kadane algoritmasını 2. denemede hatasız düzelttiğini test eder."""
    ajan = ReflexionAjani(maksimum_deneme=3)
    problem = "Bir tamsayı dizisindeki maksimum alt dizi toplamını (Maximum Subarray Sum) bulun."
    fonksiyon_adi = "max_alt_dizi_toplami"
    testler = [
        TestDurumu(girdi=[-2, 1, -3, 4, -1, 2, 1, -5, 4], beklenen=6, aciklama="Karışık dizi"),
        TestDurumu(girdi=[-1, -2, -3], beklenen=-1, aciklama="Tümü negatif dizi"),
        TestDurumu(girdi=[5, 4, -1, 7, 8], beklenen=23, aciklama="Pozitif ağırlıklı"),
    ]

    rapor = ajan.iteratif_hata_ayikla(problem, fonksiyon_adi, testler)

    assert rapor["cozuldu"] is True
    assert rapor["toplam_deneme"] == 2  # 1. deneme başarısız oldu, 2. denemede öz-eleştiriyle çözdü
    assert rapor["nihai_odul"] == 1.0


def test_gorsellestirici_pano():
    """ReflexionGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    ajan = ReflexionAjani(maksimum_deneme=3)
    testler = [
        TestDurumu(girdi=[-2, 1, -3, 4, -1, 2, 1, -5, 4], beklenen=6, aciklama="Karışık"),
        TestDurumu(girdi=[-1, -2], beklenen=-1, aciklama="Negatif"),
    ]
    rapor = ajan.iteratif_hata_ayikla("Maksimum alt dizi", "max_alt_dizi_toplami", testler)
    karsilastirma = ajan.benchmark_karsilastir()

    gorsellestirici = ReflexionGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_reflexion_pano.png")
        gorsellestirici.pano_olustur(rapor, karsilastirma, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
