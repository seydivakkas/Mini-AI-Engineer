"""
Kirchenbauer LLM Filigranlama ve Tespit Testleri (Day 118).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import math
import torch
import pytest

from src.filigran_ekleyici import KirchenbauerWatermarker
from src.filigran_tespitci import WatermarkDetector
from src.filigran_laboratuvari import FiligranLaboratuvari
from src.gorsellestirici import FiligranGorsellestirici


def test_yesil_liste_boyutu_ve_determinizm():
    """Yeşil listenin tam olarak gamma*V boyutunda olduğunu ve deterministik üretildiğini test eder."""
    watermarker = KirchenbauerWatermarker(vocab_size=1000, gamma=0.5, gizli_anahtar=42)
    yesil1 = watermarker._yesil_listeyi_uret(onceki_token=105)
    yesil2 = watermarker._yesil_listeyi_uret(onceki_token=105)

    assert len(yesil1) == 500
    assert yesil1 == yesil2  # Determinizm testi


def test_filigranli_logits_artisi():
    """filigranli_logits fonksiyonunun yeşil listedeki tokenlara tam delta eklediğini test eder."""
    watermarker = KirchenbauerWatermarker(vocab_size=100, gamma=0.5, delta=3.0, gizli_anahtar=42)
    temel_logits = torch.zeros(100)
    onceki_token = 10

    yesil_liste = watermarker._yesil_listeyi_uret(onceki_token)
    islenmis = watermarker.filigranli_logits(temel_logits, onceki_token)

    for i in range(100):
        if i in yesil_liste:
            assert math.isclose(float(islenmis[i].item()), 3.0, abs_tol=1e-5)
        else:
            assert math.isclose(float(islenmis[i].item()), 0.0, abs_tol=1e-5)


def test_token_dizisi_uretimi():
    """token_dizisi_uret fonksiyonunun doğru uzunlukta token listesi ürettiğini test eder."""
    watermarker = KirchenbauerWatermarker(vocab_size=100, gamma=0.5, delta=2.0)
    dizi = watermarker.token_dizisi_uret(baslangic_token=5, uzunluk=30, filigran_aktif=True)

    assert len(dizi) == 31  # baslangic_token + 30 token
    assert dizi[0] == 5


def test_tespitci_bos_ve_kisa_dizi():
    """WatermarkDetector sınıfının tek token veya boş dizilerde çökmeden çalıştığını test eder."""
    tespitci = WatermarkDetector(vocab_size=100, gamma=0.5)
    analiz_bos = tespitci.filigran_analizi([])
    analiz_tek = tespitci.filigran_analizi([42])

    assert analiz_bos["filigran_var_mi"] is False
    assert analiz_tek["z_skoru"] == 0.0


def test_z_skoru_formulu_ve_dogruluk():
    """Z-Skoru ve Yeşil Oran matematiksel formüllerinin doğruluğunu test eder."""
    tespitci = WatermarkDetector(vocab_size=1000, gamma=0.5, gizli_anahtar=42)

    # 100 geçişin tamamı yeşil token olan yapay bir dizi kuralım
    yapay_dizi = [1]
    for _ in range(100):
        onceki = yapay_dizi[-1]
        yesil = list(tespitci._yesil_listeyi_uret(onceki))
        yapay_dizi.append(yesil[0])

    analiz = tespitci.filigran_analizi(yapay_dizi)
    assert analiz["yesil_token_sayisi"] == 100
    assert analiz["yesil_oran"] == 1.0
    # Z = (100 - 50) / sqrt(100 * 0.5 * 0.5) = 50 / 5 = 10.0
    assert math.isclose(analiz["z_skoru"], 10.0, abs_tol=1e-3)
    assert analiz["filigran_var_mi"] is True


def test_filigranli_vs_filigransiz_z_farki():
    """Filigranlı metnin Z-Skorunun filigransız metinden belirgin şekilde yüksek olduğunu test eder."""
    watermarker = KirchenbauerWatermarker(vocab_size=500, gamma=0.5, delta=3.5, gizli_anahtar=42)
    tespitci = WatermarkDetector(vocab_size=500, gamma=0.5, gizli_anahtar=42)

    f_dizi = watermarker.token_dizisi_uret(baslangic_token=10, uzunluk=100, filigran_aktif=True)
    nf_dizi = watermarker.token_dizisi_uret(baslangic_token=10, uzunluk=100, filigran_aktif=False)

    f_analiz = tespitci.filigran_analizi(f_dizi)
    nf_analiz = tespitci.filigran_analizi(nf_dizi)

    assert f_analiz["z_skoru"] > nf_analiz["z_skoru"]
    assert f_analiz["yesil_oran"] > nf_analiz["yesil_oran"]


def test_filigran_laboratuvari_benchmark():
    """FiligranLaboratuvari sınıfının benchmark testlerini ve delta analizini hatasız yürüttüğünü test eder."""
    lab = FiligranLaboratuvari(vocab_size=500, gamma=0.5, delta=2.5, gizli_anahtar=42)
    rapor = lab.benchmark_kostur(ornek_sayisi=10, dizi_uzunlugu=50)

    assert "filigranli_ort_z" in rapor
    assert "filigransiz_ort_z" in rapor
    assert rapor["filigranli_ort_z"] > rapor["filigransiz_ort_z"]
    assert rapor["tpr_dogru_tespit_orani"] >= 80.0
    assert len(rapor["delta_degerleri"]) == len(rapor["delta_z_skorlari"])


def test_gorsellestirici_pano():
    """FiligranGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    lab = FiligranLaboratuvari(vocab_size=500, gamma=0.5, delta=2.5, gizli_anahtar=42)
    rapor = lab.benchmark_kostur(ornek_sayisi=5, dizi_uzunlugu=30)

    gorsellestirici = FiligranGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_filigran_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
