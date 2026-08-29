"""
GÜN 145: Outcome (ORM) vs Process Reward Models (PRM) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.orm_odul_modeli import OutcomeRewardModel
from src.prm_odul_modeli import ProcessRewardModel
from src.best_of_n_sirayici import BestOfNSirayici
from src.gorsellestirici import PRMvsORMGorsellestirici


def test_orm_dogru_yanit_puanlama():
    """ORM'nin doğru nihai cevabı 1.0 ile puanladığını test eder."""
    orm = OutcomeRewardModel(dogru_cevap="0.05")
    yol = {"nihai_cevap": "0.05", "adimlar": ["Adım 1", "Adım 2"]}
    res = orm.puanla(yol)

    assert res["orm_puani"] == 1.0
    assert res["nihai_cevap_dogru_mu"] is True
    assert res["ara_hata_tespit_edildi_mi"] is False


def test_orm_yanlis_yanit_puanlama():
    """ORM'nin yanlış cevabı 0.0 ile puanladığını test eder."""
    orm = OutcomeRewardModel(dogru_cevap="0.05")
    yol = {"nihai_cevap": "0.10", "adimlar": ["Adım 1"]}
    res = orm.puanla(yol)

    assert res["orm_puani"] == 0.0
    assert res["nihai_cevap_dogru_mu"] is False


def test_prm_gecerli_adim_puanlama():
    """PRM'nin doğru adımı yüksek puanla (0.98) değerlendirdiğini test eder."""
    prm = ProcessRewardModel()
    puan, etiket = prm.adim_puanla("Sopa + Top = 1.10")

    assert puan >= 0.90
    assert etiket == "dogru_adim"


def test_prm_hatali_adim_yakalama():
    """PRM'nin mantık hatası içeren adımı yakalayıp düşük puan (0.05) verdiğini test eder."""
    prm = ProcessRewardModel()
    puan, etiket = prm.adim_puanla("1.10 - 1.00 = 0.10")

    assert puan <= 0.10
    assert etiket == "hatali_adim"


def test_prm_yol_puani_ve_ilk_hata():
    """PRM'nin hatalı bir yolda ilk hata indeksini tespit ettiğini test eder."""
    prm = ProcessRewardModel()
    yol = {
        "adimlar": [
            "Sopa + Top = 1.10",
            "1.10 - 1.00 = 0.10",  # 2. adım hatalı
            "Top = 0.05",
        ],
        "nihai_cevap": "0.05",
    }
    res = prm.puanla(yol)

    assert res["ara_hata_tespit_edildi_mi"] is True
    assert res["ilk_hata_adimi"] == 2
    assert res["gecerli_yol_mu"] is False
    assert res["prm_carpim_puani"] < 0.10


def test_best_of_n_sirayici_sansli_tahmin_yakalama():
    """BestOfNSirayici'nin şanslı tahmin (lucky guess) vakalarını tespit ettiğini test eder."""
    sirayici = BestOfNSirayici(dogru_cevap="0.05")
    adaylar = [
        {
            "yol_id": "yol_1_dogru",
            "adimlar": ["Sopa + Top = 1.10", "Sopa = Top + 1.00", "2 * Top = 0.10", "Top = 0.05"],
            "nihai_cevap": "0.05",
        },
        {
            "yol_id": "yol_2_sansli",
            "adimlar": ["Sopa + Top = 1.10", "1.10 - 1.00 = 0.10", "Top = 0.05"],
            "nihai_cevap": "0.05",
        },
    ]

    res = sirayici.karsilastir_ve_sirala(adaylar)
    assert res["sansli_tahmin_sayisi"] == 1


def test_best_of_n_sirayici_en_iyi_yol_secimi():
    """BestOfNSirayici'nin PRM ile mantıksal olarak kusursuz yolu 1. sıraya aldığını test eder."""
    sirayici = BestOfNSirayici(dogru_cevap="0.05")
    adaylar = [
        {
            "yol_id": "yol_sansli",
            "adimlar": ["1.10 - 1.00 = 0.10", "Top = 0.05"],
            "nihai_cevap": "0.05",
        },
        {
            "yol_id": "yol_kusursuz",
            "adimlar": ["Sopa + Top = 1.10", "Sopa = Top + 1.00", "2 * Top = 0.10", "Top = 0.05"],
            "nihai_cevap": "0.05",
        },
    ]

    res = sirayici.karsilastir_ve_sirala(adaylar)
    assert res["prm_secimi"]["yol_id"] == "yol_kusursuz"
    assert res["prm_secimi"]["gecerli_yol_mu"] is True


def test_gorsellestirici_pano_uretme():
    """6 panelli PRM vs ORM teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    sirayici = BestOfNSirayici(dogru_cevap="0.05")
    adaylar = [
        {"yol_id": "y1", "adimlar": ["Sopa + Top = 1.10", "Top = 0.05"], "nihai_cevap": "0.05"}
    ]
    karsilastirma = sirayici.karsilastir_ve_sirala(adaylar)

    gorsellestirici = PRMvsORMGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_prm_pano.png")
        gorsellestirici.pano_olustur(karsilastirma, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
