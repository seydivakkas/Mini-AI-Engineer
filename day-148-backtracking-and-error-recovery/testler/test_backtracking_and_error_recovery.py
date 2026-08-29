"""
GÜN 148: Backtracking ve Hata Kurtarma Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.dusunce_yigini import DusunceYigini, DusunceKaresi
from src.cikmaz_sokak_tespitcisi import CikmazSokakTespitcisi
from src.geri_izleme_yoneticisi import GeriIzlemeYoneticisi
from src.gorsellestirici import BacktrackingGorsellestirici


def test_dusunce_yigini_ekleme_ve_boyut():
    """DusunceYigini'nin kare ekleme ve boyut takibini test eder."""
    yigin = DusunceYigini()
    assert yigin.bos_mu() is True

    k1 = DusunceKaresi(1, {"toplam": 1.10}, "Adım 1", kontrol_noktasi_mi=True)
    yigin.ekle(k1)

    assert yigin.boyut() == 1
    assert yigin.son_kare().adim_id == 1


def test_dusunce_yigini_rollback():
    """DusunceYigini'nin hata anında kontrol noktasına kadar geri sardığını test eder."""
    yigin = DusunceYigini()
    k0 = DusunceKaresi(0, {"sopa": None}, "Başlangıç", kontrol_noktasi_mi=True)
    k1 = DusunceKaresi(1, {"sopa": 1.10}, "Hatalı Adım", kontrol_noktasi_mi=False)
    k2 = DusunceKaresi(2, {"top": 0.10}, "Hatalı Sonuç", kontrol_noktasi_mi=False)

    yigin.ekle(k0)
    yigin.ekle(k1)
    yigin.ekle(k2)

    assert yigin.boyut() == 3
    yigin.son_gecerli_kontrol_noktasina_geri_sar("Mantık hatası")

    assert yigin.boyut() == 1
    assert yigin.son_kare().adim_id == 0


def test_cikmaz_sokak_crt_celiski():
    """CikmazSokakTespitcisi'nin CRT $0.10 çelişkisini anında tespit ettiğini test eder."""
    cikmaz_mi, neden, guven = CikmazSokakTespitcisi.denetle("O halde top = 0.10 dolar", {})
    assert cikmaz_mi is True
    assert "Çelişki" in neden
    assert guven >= 0.90


def test_cikmaz_sokak_gecerli_adim():
    """CikmazSokakTespitcisi'nin geçerli adımlara onay verdiğini test eder."""
    cikmaz_mi, neden, guven = CikmazSokakTespitcisi.denetle("Sopa = Top + 1.00 dolar", {})
    assert cikmaz_mi is False
    assert guven >= 0.90


def test_cikmaz_sokak_gecersiz_negatif():
    """CikmazSokakTespitcisi'nin negatif sonuçları çıkmaz sokak olarak işaretlediğini test eder."""
    cikmaz_mi, neden, guven = CikmazSokakTespitcisi.denetle("Kalan sayı: -5", {"sayilar": [-5, 10]})
    assert cikmaz_mi is True
    assert "Negatif" in neden


def test_geri_izleme_yoneticisi_kurtarma():
    """GeriIzlemeYoneticisi'nin çıkmaz sokakta monolog tetikleyip geri döndüğünü test eder."""
    yonetici = GeriIzlemeYoneticisi()
    adaylar = [
        {"metin": "Sopa + Top = 1.10", "yeni_durum": {"toplam": 1.10}, "kontrol_noktasi_mi": True},
        {"metin": "1.10 - 1.00 = 0.10 o halde top = 0.10", "yeni_durum": {"top": 0.10}},  # Çıkmaz sokak!
        {"metin": "Sopa = Top + 1.00", "yeni_durum": {"denklem": "sopa=top+1"}},          # Alternatif geçerli dal
        {"metin": "2 * Top = 0.10", "yeni_durum": {"fark": 0.10}},
        {"metin": "Top = 0.05", "yeni_durum": {"top": 0.05}},
    ]

    sonuc = yonetici.akil_yurut_ve_kurtar({}, adaylar)

    assert sonuc["toplam_geri_izleme_sayisi"] == 1
    assert len(sonuc["ic_monologlar"]) == 1
    assert "<think>" in sonuc["ic_monologlar"][0]
    assert sonuc["basarili_mi"] is True


def test_geri_izleme_yoneticisi_nihai_zincir():
    """GeriIzlemeYoneticisi'nin nihai zincirinde hatalı adımın yer almadığını test eder."""
    yonetici = GeriIzlemeYoneticisi()
    adaylar = [
        {"metin": "Sopa + Top = 1.10", "yeni_durum": {}, "kontrol_noktasi_mi": True},
        {"metin": "Top = 0.10", "yeni_durum": {}},  # Hatalı
        {"metin": "Sopa = Top + 1.00", "yeni_durum": {}},
        {"metin": "Top = 0.05", "yeni_durum": {}},
    ]

    sonuc = yonetici.akil_yurut_ve_kurtar({}, adaylar)
    zincir = sonuc["nihai_gecerli_zincir"]

    assert not any("Top = 0.10" in z for z in zincir)
    assert any("Top = 0.05" in z for z in zincir)


def test_gorsellestirici_pano_uretme():
    """6 panelli Backtracking teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    yonetici = GeriIzlemeYoneticisi()
    adaylar = [
        {"metin": "Adım 1", "yeni_durum": {}, "kontrol_noktasi_mi": True},
        {"metin": "Top = 0.10", "yeni_durum": {}},
        {"metin": "Düzeltilmiş Adım", "yeni_durum": {}},
    ]
    sonuc = yonetici.akil_yurut_ve_kurtar({}, adaylar)

    gorsellestirici = BacktrackingGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_backtracking_pano.png")
        gorsellestirici.pano_olustur(sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
