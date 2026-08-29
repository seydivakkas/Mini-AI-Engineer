"""
GÜN 143: Self-Consistency Sıcaklık Örneklemesi ve Entropi Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.sicaklik_ornekleyici import SicaklikOrnekleyici
from src.agirlikli_oylayici import AgirlikliOylayici
from src.entropi_belirsizlik_analizcisi import EntropiBelirsizlikAnalizcisi
from src.gorsellestirici import SelfConsistencyTemperatureGorsellestirici


def test_sicaklik_ornekleyici_greedy():
    """Sıcaklık T=0.0 olduğunda tamamen aynı deterministik yolun üretildiğini test eder."""
    ornekleyici = SicaklikOrnekleyici(tohum=42)
    yollar = ornekleyici.ornekle(n_ornek=5, sicaklik=0.0)

    assert len(yollar) == 5
    assert all(y["tahmin"] == "0.05" for y in yollar)
    assert all(y["yol_olasiligi"] == 1.0 for y in yollar)


def test_sicaklik_ornekleyici_cesitlilik():
    """Sıcaklık T=0.7 olduğunda olasılık ağırlıklı çeşitli yolların üretildiğini test eder."""
    ornekleyici = SicaklikOrnekleyici(tohum=42)
    yollar = ornekleyici.ornekle(n_ornek=5, sicaklik=0.7)

    assert len(yollar) == 5
    assert any(y["yol_olasiligi"] > 0.0 for y in yollar)
    assert all("log_olasilik" in y for y in yollar)


def test_agirlikli_oylayici_kazanan_secimi():
    """AgirlikliOylayici'nin çoğunluk ve ağırlıklı kazananı seçtiğini test eder."""
    yollar = [
        {"tahmin": "0.05", "yol_olasiligi": 0.45},
        {"tahmin": "0.05", "yol_olasiligi": 0.35},
        {"tahmin": "0.10", "yol_olasiligi": 0.20},
    ]

    sonuc = AgirlikliOylayici.oyla(yollar)
    assert sonuc["kazanan_tahmin"] == "0.05"
    assert sonuc["agirlikli_guven_skoru"] == 0.80
    assert sonuc["toplam_ornek"] == 3


def test_agirlikli_oylayici_agirlik_etkisi():
    """Yüksek olasılıklı yolun ağırlıklandırmada belirleyici olduğunu test eder."""
    yollar = [
        {"tahmin": "0.05", "yol_olasiligi": 0.70},
        {"tahmin": "0.10", "yol_olasiligi": 0.15},
        {"tahmin": "0.10", "yol_olasiligi": 0.15},
    ]

    sonuc = AgirlikliOylayici.oyla(yollar)
    assert sonuc["kazanan_tahmin"] == "0.05"  # Soft oyda 0.70 ile kazandı
    assert sonuc["kazanan_hard_tahmin"] == "0.10"  # Hard oyda 2 oy ile kazandı


def test_entropi_analizcisi_dusuk_entropi():
    """Tek bir yanıtta toplanmış dağılımın düşük Shannon entropisi ürettiğini test eder."""
    dagilim = {"0.05": 0.90, "0.10": 0.10}
    sonuc = EntropiBelirsizlikAnalizcisi.analiz_et(dagilim)

    assert sonuc["shannon_entropisi"] < 0.60
    assert sonuc["guvenli_mi"] is True
    assert "DÜŞÜK" in sonuc["belirsizlik_seviyesi"]


def test_entropi_analizcisi_yuksek_entropi():
    """Dağınık ve çelişkili dağılımın yüksek entropi (halüsinasyon riski) ürettiğini test eder."""
    dagilim = {"0.05": 0.33, "0.10": 0.33, "0.55": 0.34}
    sonuc = EntropiBelirsizlikAnalizcisi.analiz_et(dagilim)

    assert sonuc["shannon_entropisi"] > 1.20
    assert sonuc["guvenli_mi"] is False
    assert "YÜKSEK" in sonuc["belirsizlik_seviyesi"]


def test_sicaklik_tarama_trendleri():
    """Sıcaklık arttıkça entropinin yükseldiğini test eder."""
    t_degerleri = [0.0, 0.3, 0.7, 1.2]
    entropiler = [0.0, 0.25, 0.46, 1.35]

    assert entropiler[0] == 0.0
    assert entropiler[-1] > entropiler[2] > entropiler[1]


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun başarıyla PNG olarak kaydedildiğini test eder."""
    oylama = {
        "kazanan_tahmin": "0.05",
        "hard_oy_dagilimi": {"0.05": 4, "0.10": 1},
        "agirlikli_oy_dagilimi": {"0.05": 0.88, "0.10": 0.12},
    }
    entropi = {
        "shannon_entropisi": 0.52,
        "gini_kirliligi": 0.21,
        "maksimum_olasilik": 0.88,
        "belirsizlik_seviyesi": "DÜŞÜK_BELİRSİZLİK",
    }
    tarama = {
        "sicakliklar": [0.0, 0.3, 0.7, 1.2],
        "entropiler": [0.0, 0.25, 0.46, 1.35],
        "dogruluklar": [60.0, 80.0, 100.0, 70.0],
    }

    gorsellestirici = SelfConsistencyTemperatureGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_sc_temp_pano.png")
        gorsellestirici.pano_olustur(oylama, entropi, tarama, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
