"""
Evol-Instruct & UltraFeedback Sentetik Veri Testleri (Day 116).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.evrim_operatorleri import EvolInstructMotoru
from src.kalite_filtresi import SentetikKaliteFiltresi
from src.ultrafeedback_motoru import UltraFeedbackPuanlayici
from src.sentetik_laboratuvar import SentetikVeriLaboratuvari
from src.gorsellestirici import SentetikVeriGorsellestirici


def test_in_depth_kisit_ekle():
    """in_depth_kisit_ekle operatörünün isteme kısıt eklediğini test eder."""
    motor = EvolInstructMotoru(seed=42)
    tohum = "Bir Python fonksiyonu yazın."
    evrilmis = motor.in_depth_kisit_ekle(tohum)

    assert "[KISIT]" in evrilmis
    assert len(evrilmis) > len(tohum)


def test_in_depth_derinlestir_ve_somutlastir():
    """Derinleştirme ve somutlaştırma operatörlerinin doğru çalıştığını test eder."""
    motor = EvolInstructMotoru(seed=42)
    tohum = "Kayıp fonksiyonu nedir?"

    derin = motor.in_depth_derinlestir(tohum)
    somut = motor.in_depth_somutlastir(tohum)

    assert "[DERİNLEŞTİRME]" in derin
    assert "[SOMUTLAŞTIRMA]" in somut


def test_in_breadth_mutasyon():
    """Genişlemesine mutasyon operatörünün çalıştığını test eder."""
    motor = EvolInstructMotoru(seed=42)
    tohum = "İlişkisel veritabanı indekslemesi."
    mutasyon = motor.in_breadth_mutasyon(tohum)

    assert "[MUTASYON ÇEŞİTLİLİK]" in mutasyon


def test_jaccard_benzerlik():
    """jaccard_benzerlik fonksiyonunun doğru oran ürettiğini test eder."""
    filtre = SentetikKaliteFiltresi()
    s1 = "Python ile makine öğrenimi modeli eğitimi"
    s2 = "Python ile derin öğrenme modeli eğitimi"

    benzerlik = filtre.jaccard_benzerlik(s1, s2)
    assert 0.0 < benzerlik < 1.0
    assert filtre.jaccard_benzerlik(s1, s1) == 1.0


def test_karmasiklik_skoru():
    """karmasiklik_skoru fonksiyonunun teknik sözcük içeren uzun isteme daha yüksek puan verdiğini test eder."""
    filtre = SentetikKaliteFiltresi()
    basit = "Sıralama yapın."
    gelismis = "Python'da zaman karmaşıklığı ve bellek optimizasyonu içeren asimptotik algoritma kanıtı [KISIT]: O(1) bellek."

    skor_basit = filtre.karmasiklik_skoru(basit)
    skor_gelismis = filtre.karmasiklik_skoru(gelismis)

    assert skor_gelismis > skor_basit


def test_gecerlilik_elemesi():
    """gecerlilik_elemesi fonksiyonunun geçerli evrimi kabul edip yetersiz/kısa istemi elediğini test eder."""
    filtre = SentetikKaliteFiltresi()
    tohum = "Python'da sıralama fonksiyonu yazın."
    iyi_evrim = "Python'da zaman karmaşıklığı ve bellek optimizasyonu içeren asimptotik algoritma kanıtı [KISIT]: O(1) bellek."
    kopya = "Python'da sıralama fonksiyonu yazın."

    kabul, _ = filtre.gecerlilik_elemesi(tohum, iyi_evrim)
    ret, sebep = filtre.gecerlilik_elemesi(tohum, kopya)

    assert kabul is True
    assert ret is False
    assert "RET" in sebep


def test_ultrafeedback_puanlama_ve_cift():
    """UltraFeedbackPuanlayici modülünün adayları puanlayıp chosen > rejected ürettiğini test eder."""
    puanlayici = UltraFeedbackPuanlayici(seed=42)
    prompt = "Bir sıralama algoritması yazın."
    adaylar = [
        "Adım 1: Zaman karmaşıklığı O(1) olan Quicksort. ```python\ndef qsort(): pass\n```",
        "Basit sıralama için sort() kullanın.",
        "Sıralama yapılabilir.",
    ]

    cift = puanlayici.tercih_cifti_uret(prompt, adaylar)

    assert "chosen" in cift
    assert "rejected" in cift
    assert cift["chosen_skor"] >= cift["rejected_skor"]
    assert "elestiri" in cift


def test_sentetik_laboratuvar_ve_gorsellestirici():
    """SentetikVeriLaboratuvari ve Görselleştirici entegrasyonunu test eder."""
    lab = SentetikVeriLaboratuvari(seed=42)
    rapor = lab.evrim_laboratuvarini_kostur(nesil_sayisi=2)

    assert "ortalama_skorlar" in rapor
    assert len(rapor["ortalama_skorlar"]) == 3
    assert rapor["toplam_cift_sayisi"] > 0

    gorsellestirici = SentetikVeriGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_evol_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
