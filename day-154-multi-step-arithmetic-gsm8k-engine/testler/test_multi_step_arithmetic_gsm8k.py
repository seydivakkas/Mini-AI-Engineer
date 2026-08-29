"""
GÜN 154: GSM8K & MATH Çok Adımlı Matematiksel Akıl Yürütme (PAL) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.aritmetik_ayristirici import AritmetikAyristirici
from src.pal_kod_ureticisi import PALKodUreticisi
from src.gsm8k_yurutucu import GSM8KYurutucu
from src.gorsellestirici import GSM8KGorsellestirici


def test_aritmetik_ayristirici_sayi_tespiti():
    """Aritmetik problem metninden tüm sayıların doğru ayıklandığını test eder."""
    metin = "Ahmet 15 elmanın 6 tanesini yedi ve 4.5 TL harcadı."
    sonuc = AritmetikAyristirici.ayristir(metin)

    assert 15 in sonuc["tespit_edilen_sayilar"]
    assert 6 in sonuc["tespit_edilen_sayilar"]
    assert 4.5 in sonuc["tespit_edilen_sayilar"]
    assert sonuc["sayi_adedi"] == 3


def test_pal_kod_ureticisi_elma_problemi():
    """PAL kod üreticisinin elma problemi için çalıştırılabilir kod ürettiğini test eder."""
    ureticisi = PALKodUreticisi()
    problem = "Ayşe'nin 15 elması vardı. 3 arkadaşının her birine 2'şer elma verdi."
    kod_bilgisi = ureticisi.kod_uret("p1", problem)

    assert "def solution():" in kod_bilgisi["python_kodu"]
    assert "toplam_elma = 15" in kod_bilgisi["python_kodu"]


def test_pal_kod_ureticisi_firin_problemi():
    """PAL kod üreticisinin fırın problemi için doğru Python fonksiyonu ürettiğini test eder."""
    ureticisi = PALKodUreticisi()
    problem = "Bir fırın sabah 120 ekmek, öğlen 80 ekmek üretti."
    kod_bilgisi = ureticisi.kod_uret("p2", problem)

    assert "toplam_uretim = 120 + 80" in kod_bilgisi["python_kodu"]
    assert "return elde_edilen_gelir" in kod_bilgisi["python_kodu"]


def test_gsm8k_yurutucu_kodu_calistir_basarili():
    """İzole Python yürütücüsünün solution() fonksiyonunu hatasız çalıştırdığını test eder."""
    kod = "def solution():\n    return (10 * 5) - 8\n"
    sonuc = GSM8KYurutucu.kodu_calistir(kod)

    assert sonuc["basarili_mi"] is True
    assert sonuc["sonuc"] == 42
    assert sonuc["hata"] is None


def test_gsm8k_yurutucu_hatali_kod_yakalama():
    """Sözdizimi veya çalışma zamanı hatası olan kodların yakalandığını test eder."""
    hatali_kod = "def solution():\n    return 10 / 0\n"
    sonuc = GSM8KYurutucu.kodu_calistir(hatali_kod)

    assert sonuc["basarili_mi"] is False
    assert sonuc["sonuc"] is None
    assert "division by zero" in sonuc["hata"]


def test_gsm8k_yurutucu_cozum_karsilastir():
    """PAL sonucu ile doğrudan CoT sonucunun karşılaştırıldığını test eder."""
    kod = "def solution():\n    return 15 - (3 * 2) - 4.5\n"
    karsilastirma = GSM8KYurutucu.cozum_karsilastir(
        problem_adi="Elma",
        problem_metni="15 elma...",
        pal_kodu=kod,
        beklenen_sonuc=4.5,
        raw_cot_tahmini=6.0, # Hatalı CoT
    )

    assert karsilastirma["pal_dogru_mu"] is True
    assert karsilastirma["raw_cot_dogru_mu"] is False
    assert karsilastirma["pal_sonucu"] == 4.5


def test_pal_coklu_adim_dogrulugu():
    """Çok adımlı vergi ve indirim hesabının PAL ile %100 doğruluğunu test eder."""
    ureticisi = PALKodUreticisi()
    kod_bilgisi = ureticisi.kod_uret("p_vergi", "250 TL ürün %20 indirim ve %18 KDV")
    sonuc = GSM8KYurutucu.kodu_calistir(kod_bilgisi["python_kodu"])

    # 250 * 0.8 = 200; 200 * 1.18 = 236.0
    assert sonuc["basarili_mi"] is True
    assert sonuc["sonuc"] == 236.0


def test_gorsellestirici_pano_uretme():
    """6 panelli GSM8K ve PAL teşhis panosunun PNG olarak kaydedildiğini test eder."""
    karsilastirma_listesi = [
        {
            "problem_adi": "Elma Problemi",
            "problem_metni": "Ayşe'nin 15 elması vardı...",
            "beklenen_sonuc": 4.5,
            "pal_kodu": "def solution():\n    return 4.5\n",
            "pal_sonucu": 4.5,
            "pal_dogru_mu": True,
            "raw_cot_tahmini": 6.0,
            "raw_cot_dogru_mu": False,
            "calisma_suresi_ms": 0.15,
        }
    ]
    gorsellestirici = GSM8KGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_gsm8k_pano.png")
        gorsellestirici.pano_olustur(karsilastirma_listesi, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
