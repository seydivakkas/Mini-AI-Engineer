"""
GÜN 149: Self-Verification ve Eleştiri Döngüsü Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.aktor_cozucu import AktorCozucu
from src.elestirmen_dogrulayici import ElestirmenDogrulayici
from src.dogrulama_dongusu_yoneticisi import DogrulamaDongusuYoneticisi
from src.gorsellestirici import SelfVerificationGorsellestirici


def test_aktor_cozucu_ilk_taslak():
    """AktorCozucu'nün ilk taslak çözümünü ve güven skorunu test eder."""
    aktor = AktorCozucu()
    problem = {"tur": "moduler_aritmetik"}
    cozum = aktor.ilk_cozumu_uret(problem)

    assert "aday_x" in cozum
    assert "dusunce_zinciri" in cozum
    assert 0.0 <= cozum["guven_skoru"] <= 1.0


def test_elestirmen_hatali_ters_saglama():
    """ElestirmenDogrulayici'nin hatalı x=2 adayını reddettiğini test eder."""
    elestirmen = ElestirmenDogrulayici()
    problem = {"tur": "moduler_aritmetik"}
    hatali_cozum = {"aday_x": 2}

    sonuc = elestirmen.ters_saglama_yap(problem, hatali_cozum)
    assert sonuc["dogrulandi_mi"] is False
    assert sonuc["hesaplanan_deger"] == 3
    assert sonuc["beklenen_deger"] == 2
    assert "Ters Sağlama Başarısız" in sonuc["hata_mesaji"]


def test_elestirmen_dogru_ters_saglama():
    """ElestirmenDogrulayici'nin doğru x=0 adayını onayladığını test eder."""
    elestirmen = ElestirmenDogrulayici()
    problem = {"tur": "moduler_aritmetik"}
    dogru_cozum = {"aday_x": 0}

    sonuc = elestirmen.ters_saglama_yap(problem, dogru_cozum)
    assert sonuc["dogrulandi_mi"] is True
    assert sonuc["hesaplanan_deger"] == 2
    assert sonuc["beklenen_deger"] == 2
    assert sonuc["kesinlik_skoru"] == 1.0


def test_aktor_elestiriye_gore_rafinasyon():
    """AktorCozucu'nün eleştiri raporunu kullanarak doğru kökü bulduğunu test eder."""
    aktor = AktorCozucu()
    problem = {"tur": "moduler_aritmetik"}
    onceki = {"aday_x": 2, "tur_sayisi": 1}
    elestiri = {"hata_mesaji": "3*(2) + 7 = 13 mod 5 = 3 != 2"}

    yeni = aktor.elestiriye_gore_rafine_et(problem, onceki, elestiri)
    assert yeni["aday_x"] == 0
    assert yeni["tur_sayisi"] == 2
    assert yeni["guven_skoru"] > 0.90


def test_dogrulama_dongusu_tam_akis():
    """DogrulamaDongusuYoneticisi'nin 2. turda çözümü kusursuz doğruladığını test eder."""
    yonetici = DogrulamaDongusuYoneticisi()
    problem = {"tur": "moduler_aritmetik"}
    sonuc = yonetici.calistir(problem)

    assert sonuc["toplam_tur_sayisi"] == 2
    assert sonuc["basarili_dogrulandi_mi"] is True
    assert sonuc["nihai_cozum"] == 0


def test_dogrulama_dongusu_genel_denklem():
    """DogrulamaDongusuYoneticisi'nin genel cebirsel denklemlerde de çalıştığını test eder."""
    yonetici = DogrulamaDongusuYoneticisi()
    problem = {"tur": "cebirsel_denklem"}
    sonuc = yonetici.calistir(problem)

    assert sonuc["basarili_dogrulandi_mi"] is True
    assert sonuc["nihai_cozum"] == 5


def test_dogrulama_dongusu_kesinlik_skoru():
    """Nihai turun kesinlik ve doğrulama durumunu test eder."""
    yonetici = DogrulamaDongusuYoneticisi()
    problem = {"tur": "moduler_aritmetik"}
    sonuc = yonetici.calistir(problem)

    son_tur = sonuc["dongu_kayitlari"][-1]
    assert son_tur["dogrulandi_mi"] is True
    assert son_tur["guven_skoru"] >= 0.95


def test_gorsellestirici_pano_uretme():
    """6 panelli Self-Verification teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    yonetici = DogrulamaDongusuYoneticisi()
    problem = {"tur": "moduler_aritmetik"}
    sonuc = yonetici.calistir(problem)

    gorsellestirici = SelfVerificationGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_verification_pano.png")
        gorsellestirici.pano_olustur(sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
