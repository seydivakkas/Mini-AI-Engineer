"""
GÜN 142: Chain-of-Thought (<think>) ve Self-Consistency Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.dusunce_tokenizatoru import DusunceTokenizatoru
from src.cot_akil_yurutucu import COTAkilYurutucu
from src.self_consistency_birlestirici import SelfConsistencyBirlestirici
from src.gorsellestirici import COTSelfConsistencyGorsellestirici


def test_dusunce_tokenizatoru_ayristir():
    """DusunceTokenizatoru'nün <think> düşünce bloğunu ve nihai yanıtı başarıyla ayırdığını test eder."""
    tokenizator = DusunceTokenizatoru()
    metin = "<think>\nAdım 1: Denklemi kur.\nAdım 2: Çöz.\n</think>\nNihai Yanıt: 5 cent."

    sonuc = tokenizator.ayristir(metin)
    assert sonuc["dusunce_mevcut_mu"] is True
    assert "Adım 1: Denklemi kur." in sonuc["dusunce_metni"]
    assert sonuc["nihai_yanit"] == "Nihai Yanıt: 5 cent."
    assert len(sonuc["adimlar"]) == 2


def test_dusunce_tokenizatoru_birlestir():
    """DusunceTokenizatoru'nün standart <think> formatında birleştirme yaptığını test eder."""
    tokenizator = DusunceTokenizatoru()
    bilesik = tokenizator.birlestir("Düşünce adımı", "Doğru yanıt")

    assert bilesik.startswith("<think>")
    assert "</think>" in bilesik
    assert bilesik.endswith("Doğru yanıt")


def test_dusunce_tokenizatoru_token_dagilimi():
    """Token dağılımı hesaplayıcısının prompt, düşünce ve yanıt tokenlerini doğru saydığını test eder."""
    tokenizator = DusunceTokenizatoru()
    dagilim = tokenizator.token_dagilimi_hesapla(
        prompt="Soru metni",
        dusunce_metni="Adım 1 Adım 2",
        nihai_yanit="Sonuç",
    )

    assert dagilim["prompt_token_sayisi"] == 2
    assert dagilim["dusunce_token_sayisi"] == 6  # 4 kelime + 2 özel token
    assert dagilim["nihai_yanit_token_sayisi"] == 1
    assert dagilim["toplam_token_sayisi"] == 9


def test_cot_akil_yurutucu_coklu_yol():
    """COTAkilYurutucu'nün K adet bağımsız akıl yürütme yolu ürettiğini test eder."""
    yurutucu = COTAkilYurutucu()
    yollar = yurutucu.ornekle_coklu_yol("sopave_top", "Sopa top sorusu...", k=5)

    assert len(yollar) == 5
    assert all("dusunce_metni" in y for y in yollar)
    assert all("nihai_yanit" in y for y in yollar)
    assert all("tahmin" in y for y in yollar)


def test_self_consistency_birlestirici_cogunluk_oyu():
    """SelfConsistencyBirlestirici'nin 4/5 çoğunluk oyuyla 0.05 tahminini seçtiğini test eder."""
    yurutucu = COTAkilYurutucu()
    yollar = yurutucu.ornekle_coklu_yol("sopave_top", "Sopa top sorusu...", k=5)

    sonuc = SelfConsistencyBirlestirici.birlestir(yollar)
    assert sonuc["kazanan_tahmin"] == "0.05"
    assert sonuc["kazanan_oy"] == 4
    assert sonuc["toplam_oy"] == 5
    assert sonuc["konsensus_skoru"] == 0.80
    assert sonuc["guvenli_mi"] is True


def test_self_consistency_sapan_yol_tespiti():
    """SelfConsistencyBirlestirici'nin azınlıkta kalan sapan yolu (0.10) tespit ettiğini test eder."""
    yurutucu = COTAkilYurutucu()
    yollar = yurutucu.ornekle_coklu_yol("sopave_top", "Sopa top sorusu...", k=5)

    sonuc = SelfConsistencyBirlestirici.birlestir(yollar)
    assert sonuc["sapan_yol_sayisi"] == 1
    assert "0.10" in sonuc["oy_dagilimi"]


def test_self_consistency_nilufer_golu():
    """Nilüfer gölü probleminde çoğunluk oyunun 47 gün olduğunu test eder."""
    yurutucu = COTAkilYurutucu()
    yollar = yurutucu.ornekle_coklu_yol("nilufer_golu", "Nilüfer gölü...", k=5)

    sonuc = SelfConsistencyBirlestirici.birlestir(yollar)
    assert sonuc["kazanan_tahmin"] == "47"
    assert sonuc["konsensus_skoru"] >= 0.80


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun başarıyla PNG dosyası olarak kaydedildiğini test eder."""
    yurutucu = COTAkilYurutucu()
    yollar = yurutucu.ornekle_coklu_yol("sopave_top", "Sopa top...", k=5)
    konsensus = SelfConsistencyBirlestirici.birlestir(yollar)
    token_dagilimi = yollar[0]["token_bilgisi"]

    gorsellestirici = COTSelfConsistencyGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_cot_pano.png")
        gorsellestirici.pano_olustur(konsensus, yollar, token_dagilimi, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
