"""
Multimodal Omni Benchmark Suite Test Paketi (Day 180 - FAZ 9 BÜYÜK FİNALİ).
8 adet kapsamlı PyTest birim testi.
"""

import sys
import os
import tempfile
import pytest

# Proje dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mme_degerlendirici import MMEDegerlendirici
from src.mmbench_degerlendirici import MMBenchDegerlendirici
from src.mathvista_degerlendirici import MathVistaDegerlendirici
from src.omni_karsilastirici import OmniBenchmarkMerkezi
from src.gorsellestirici import MultimodalBenchmarkGorsellestirici


def test_mme_alt_gorev_skoru_hesaplama():
    """1. MME alt görev soru çifti doğruluğu (Acc ve Acc+) hesaplama testi."""
    # 4 imaj:
    # İmaj 1: Q1=Doğru, Q2=Doğru (Tam doğru)
    # İmaj 2: Q1=Doğru, Q2=Yanlış
    # İmaj 3: Q1=Yanlış, Q2=Doğru
    # İmaj 4: Q1=Yanlış, Q2=Yanlış
    ciftler = [(True, True), (True, False), (False, True), (False, False)]
    res = MMEDegerlendirici.alt_gorev_skoru_hesapla(ciftler)

    # Toplam 8 sorudan 4'ü doğru -> Acc = %50.0
    # 4 imajdan 1'i tam doğru -> Acc+ = %25.0
    # Toplam skor = 50 + 25 = 75.0
    assert res["acc"] == 50.0, "Acc değeri %50.0 olmalıdır."
    assert res["acc_plus"] == 25.0, "Acc+ değeri %25.0 olmalıdır."
    assert res["skor"] == 75.0, "Alt görev skoru 75.0 olmalıdır."


def test_mme_tam_degerlendirme_skor_araligi():
    """2. 14 görevli MME tam değerlendirme ve skor sınırları testi."""
    evaluator = MMEDegerlendirici()
    rapor = evaluator.ornek_model_mme_raporu("Mini-Omni-v1")

    assert 0 <= rapor["perception_skoru"] <= 2000.0, "Perception skoru 0..2000 arasında olmalıdır."
    assert 0 <= rapor["cognition_skoru"] <= 800.0, "Cognition skoru 0..800 arasında olmalıdır."
    assert 0 <= rapor["toplam_mme_skoru"] <= 2800.0, "Toplam MME skoru 0..2800 arasında olmalıdır."
    assert len(rapor["perception_detay"]) == 10, "10 perception alt görevi olmalıdır."
    assert len(rapor["cognition_detay"]) == 4, "4 cognition alt görevi olmalıdır."


def test_mmbench_circular_eval_tek_soru():
    """3. MMBench CircularEval tek soru 4 turlu permütasyon testi."""
    # Durum 1: 4 turun tamamında doğru seçeneği bulan model
    turlar_dogru = [0, 1, 2, 3]
    dogru_idx = [0, 1, 2, 3]
    res1 = MMBenchDegerlendirici.circular_eval_sorusu_degerlendir(turlar_dogru, dogru_idx)
    assert res1["tam_dogru"] is True
    assert res1["tek_tur_dogruluk_orani"] == 1.0

    # Durum 2: 3 turda doğru bulan ama 1 turda pozisyon önyargısıyla şaşıran model
    turlar_yanlis = [0, 1, 2, 0]  # 4. turda 3 yerine 0 seçti
    res2 = MMBenchDegerlendirici.circular_eval_sorusu_degerlendir(turlar_yanlis, dogru_idx)
    assert res2["tam_dogru"] is False
    assert res2["tek_tur_dogruluk_orani"] == 0.75


def test_mmbench_toplu_degerlendirme_metrikleri():
    """4. MMBench toplu kategori değerlendirmesi ve tutarlılık testi."""
    evaluator = MMBenchDegerlendirici()
    rapor = evaluator.ornek_model_mmbench_raporu("Mini-Omni-v1")

    assert 0 <= rapor["genel_circular_acc"] <= 100.0
    assert 0 <= rapor["genel_vanilla_acc"] <= 100.0
    assert rapor["genel_circular_acc"] <= rapor["genel_vanilla_acc"], "CircularEval acc standart acc'den küçük veya eşit olmalıdır."
    assert "fine_grained_perception" in rapor["kategori_detaylari"]


def test_mathvista_matematiksel_cevap_karsilastirma():
    """5. MathVista matematiksel tolerans ve string parsing doğrulama testi."""
    # Sayısal tolerans testi
    assert MathVistaDegerlendirici.matematiksel_cevap_karsilastir("42.001", "42.0", soru_tipi="sayisal", tolerans=0.01) is True
    assert MathVistaDegerlendirici.matematiksel_cevap_karsilastir("45.5", "42.0", soru_tipi="sayisal", tolerans=0.01) is False

    # Metinsel / seçenekli soru testi
    assert MathVistaDegerlendirici.matematiksel_cevap_karsilastir("A", "a", soru_tipi="secenekli") is True
    assert MathVistaDegerlendirici.matematiksel_cevap_karsilastir("Parabol", "Hiperbol", soru_tipi="metinsel") is False


def test_mathvista_toplu_degerlendirme():
    """6. MathVista alt disiplinler ve genel skor derleme testi."""
    evaluator = MathVistaDegerlendirici()
    rapor = evaluator.ornek_model_mathvista_raporu("Mini-Omni-v1")

    assert 0 <= rapor["genel_mathvista_skoru"] <= 100.0
    assert "geometry_reasoning" in rapor["alan_bazli_performans"]
    assert "function_plots_and_calculus" in rapor["alan_bazli_performans"]
    assert rapor["toplam_soru"] > 0


def test_omni_benchmark_liderlik_tablosu():
    """7. Omni Benchmark Merkezi liderlik tablosu sıralaması ve ağırlık testi."""
    merkez = OmniBenchmarkMerkezi()
    rapor = merkez.tum_modelleri_karsilastir()

    assert len(rapor["liderlik_tablosu"]) == 6, "6 model değerlendirilmelidir."
    assert rapor["liderlik_tablosu"][0]["siralama"] == 1

    # Sıralama kontrolü (Omni skoruna göre azalan sırada olmalı)
    skorlar = [m["omni_score"] for m in rapor["liderlik_tablosu"]]
    assert skorlar == sorted(skorlar, reverse=True), "Liderlik tablosu Omni-Score'a göre sıralı olmalıdır."


def test_gorsellestirme_cikti_dosyasi():
    """8. 6 panelli FAZ 9 Büyük Final teşhis panosu görselleştirme testi."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kayit_yolu = os.path.join(tmpdir, "test_omni_benchmark.png")
        merkez = OmniBenchmarkMerkezi()
        rapor = merkez.tum_modelleri_karsilastir()

        gorsellestirici = MultimodalBenchmarkGorsellestirici(dpi=100)
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit_yolu)

        assert os.path.exists(kayit_yolu), "Görselleştirme dosyası kaydedilmiş olmalıdır."
        assert os.path.getsize(kayit_yolu) > 1000, "Dosya boyutu geçerli olmalıdır."
