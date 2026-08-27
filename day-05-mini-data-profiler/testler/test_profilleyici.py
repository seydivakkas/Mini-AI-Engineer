"""Mini Veri Profilleyici Birim Testleri.

Bu dosya; veri boyutu hesaplamalarını, eksiklik tespitini, çarpıklık ve basıklık
ölçümlerini, sıfır varyans alarmlarını ve anlamsal tip çıkarımlarını test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Proje kök dizinini ekler
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.veri_profilleyici import MiniVeriProfilleyici
from src.rapor_olusturucu import ProfilRaporOlusturucu


def test_gecersiz_girdi_hatalari():
    """Geçersiz tip veya boş DataFrame verildiğinde uygun hataların fırlatıldığını denetler."""
    profilleyici = MiniVeriProfilleyici()
    with pytest.raises(TypeError):
        profilleyici.profili_cikar("gecersiz_tip")  # type: ignore

    with pytest.raises(ValueError):
        profilleyici.profili_cikar(pd.DataFrame())


def test_temel_boyut_ve_eksik_hesaplama():
    """Satır, sütun ve eksik veri oranlarının doğruluğunu test eder."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": ["x", "y", "z", None]
    })
    profilleyici = MiniVeriProfilleyici()
    profil = profilleyici.profili_cikar(df)

    assert profil.satir_sayisi == 4
    assert profil.sutun_sayisi == 2
    assert profil.toplam_hucre == 8
    assert profil.toplam_eksik_hucre == 2
    assert np.isclose(profil.genel_eksik_orani, 0.25)


def test_sabit_sutun_sifir_varyans_uyarisi():
    """Tüm değerleri aynı olan sütunda sıfır varyans uyarısı üretildiğini test eder."""
    df = pd.DataFrame({
        "sabit": [42, 42, 42, 42, 42]
    })
    profilleyici = MiniVeriProfilleyici()
    profil = profilleyici.profili_cikar(df)

    uyarilar = profil.sutunlar["sabit"].uyarilar
    assert any("Sıfır Varyans" in u for u in uyarilar)


def test_yuksek_carpiklik_uyarisi():
    """Aşırı asimetrik (sağa çarpık) dağılımda çarpıklık uyarısının tetiklendiğini test eder."""
    # Sağa aşırı çarpık veri: çoğu küçük, birkaçı çok büyük
    np.random.seed(42)
    carpik_veri = np.concatenate([np.ones(100), [1000.0, 2000.0, 5000.0]])
    df = pd.DataFrame({"deger": carpik_veri})

    profilleyici = MiniVeriProfilleyici(yuksek_carpiklik_esigi=1.5)
    profil = profilleyici.profili_cikar(df)

    assert profil.sutunlar["deger"].istatistikler["carpiklik"] > 1.5
    assert any("Yüksek çarpıklık" in u for u in profil.sutunlar["deger"].uyarilar)


def test_anlamsal_tip_benzersiz_kimlik():
    """Kardinalitesi %95'in üzerinde olan sütunun benzersiz kimlik olarak sınıflandırıldığını doğrular."""
    df = pd.DataFrame({
        "id": [f"KIMLIK_{i}" for i in range(100)],
        "puan": np.random.randn(100)
    })
    profilleyici = MiniVeriProfilleyici()
    profil = profilleyici.profili_cikar(df)

    assert profil.sutunlar["id"].anlamsal_tip == "benzersiz_kimlik"
    assert profil.sutunlar["puan"].anlamsal_tip == "sayisal_surekli"


def test_markdown_rapor_metni():
    """Markdown raporlama fonksiyonunun geçerli başlıklar ve tablolar ürettiğini doğrular."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    profilleyici = MiniVeriProfilleyici()
    profil = profilleyici.profili_cikar(df)

    md = ProfilRaporOlusturucu.markdown_raporu_uret(profil)
    assert "# 📊 Veri Seti Profilleme Raporu" in md
    assert "| `x` |" in md
