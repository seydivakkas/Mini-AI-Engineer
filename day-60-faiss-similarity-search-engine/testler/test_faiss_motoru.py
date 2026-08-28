"""
Day 60: FAISS ile Milyonluk Vektör İndeksleme ve Benzerlik Arama Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.indeks_motoru import FAISSIndeksMotoru, IndeksTuru
from src.vektor_benchmark import VektorBenchmarkRunner
from src.gorsellestirici import FAISSGorsellestirici


@pytest.fixture
def ornek_vektorler():
    np.random.seed(42)
    dim = 32
    num_vectors = 500
    vektorler = np.random.randn(num_vectors, dim).astype(np.float32)
    vektorler = vektorler / np.linalg.norm(vektorler, axis=1, keepdims=True)
    return np.ascontiguousarray(vektorler)


def test_faiss_flat_ip_arama(ornek_vektorler):
    """IndexFlatIP ile arama yapıldığında aynı vektörün en yüksek skorla (1.0) 1. sırada bulunduğunu test eder."""
    dim = ornek_vektorler.shape[1]
    motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.FLAT_IP)
    motor.egit_ve_ekle(ornek_vektorler)

    sorgu = ornek_vektorler[0:1]
    skorlar, indeksler, ms = motor.ara(sorgu, top_k=5)

    assert indeksler[0, 0] == 0
    assert abs(skorlar[0, 0] - 1.0) < 1e-4
    assert ms >= 0.0


def test_faiss_ivf_flat_egitim_ve_arama(ornek_vektorler):
    """IndexIVFFlat indeksinin başarıyla eğitildiğini, eklendiğini ve arama yaptığını test eder."""
    dim = ornek_vektorler.shape[1]
    motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.IVF_FLAT, nlist=10)
    motor.egit_ve_ekle(ornek_vektorler)

    sorgular = ornek_vektorler[:5]
    skorlar, indeksler, ms = motor.ara(sorgular, top_k=3, nprobe=4)

    assert skorlar.shape == (5, 3)
    assert indeksler.shape == (5, 3)
    assert motor.toplam_vektor == len(ornek_vektorler)


def test_faiss_hnsw_flat_arama(ornek_vektorler):
    """IndexHNSWFlat graf indeksinin oluşturulduğunu ve efSearch ile arandığını test eder."""
    dim = ornek_vektorler.shape[1]
    motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.HNSW_FLAT, M=16, efConstruction=32)
    motor.egit_ve_ekle(ornek_vektorler)

    sorgu = ornek_vektorler[:2]
    skorlar, indeksler, ms = motor.ara(sorgu, top_k=5, ef_search=32)

    assert skorlar.shape == (2, 5)
    assert indeksler.shape == (2, 5)
    assert indeksler[0, 0] == 0


def test_recall_hesaplama_fonksiyonu():
    """Ground truth ve ANN indeksleri eşit olduğunda Recall@k'nın %100 döndüğünü test eder."""
    gt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    ann = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    recall = VektorBenchmarkRunner.recall_hesapla(ann, gt, k=4)
    assert recall == 100.0

    ann_yarim = np.array([[0, 1, 99, 98], [4, 5, 6, 7]])
    recall_yarim = VektorBenchmarkRunner.recall_hesapla(ann_yarim, gt, k=4)
    assert recall_yarim == 75.0


def test_indeks_serilestirme_ve_yukleme(ornek_vektorler, tmp_path):
    """FAISS indeksinin diske yazılıp tekrar başarıyla okunduğunu test eder."""
    dim = ornek_vektorler.shape[1]
    motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.FLAT_IP)
    motor.egit_ve_ekle(ornek_vektorler)

    dosya = str(tmp_path / "test_index.faiss")
    motor.indeksi_kaydet(dosya)
    assert os.path.exists(dosya)

    yuklenen = FAISSIndeksMotoru.indeksi_yukle(dosya)
    assert yuklenen.toplam_vektor == len(ornek_vektorler)


def test_vektor_benchmark_runner(ornek_vektorler):
    """VektorBenchmarkRunner'ın tüm indeksleri sorunsuz benchmark ettiğini test eder."""
    sorgular = ornek_vektorler[:10]
    sonuclar = VektorBenchmarkRunner.calistir_karsilastirma(ornek_vektorler, sorgular, top_k=5)

    assert "IndexFlatIP (Exact)" in sonuclar
    assert any("IndexIVFFlat" in k for k in sonuclar)
    assert any("IndexHNSWFlat" in k for k in sonuclar)


def test_gorsellestirici_panel_cizimi(tmp_path):
    """6 panelli FAISS görselleştiricisinin geçerli bir PNG dosyası ürettiğini test eder."""
    sonuclar = {
        "IndexFlatIP (Exact)": {
            "qps": 2000.0, "recall": 100.0, "tekil_sorgu_ms": 0.5,
            "build_suresi_s": 0.01, "bellek_tahmini_mb": 50.0
        },
        "IndexHNSWFlat (ef=32)": {
            "qps": 45000.0, "recall": 98.2, "tekil_sorgu_ms": 0.02,
            "build_suresi_s": 0.8, "bellek_tahmini_mb": 80.0
        }
    }
    hedef = str(tmp_path / "test_faiss_paneli.png")
    cikis = FAISSGorsellestirici.panel_ciz(sonuclar, num_vectors=1000, dim=32, hedef_path=hedef)

    assert os.path.exists(cikis)
    assert os.path.getsize(cikis) > 1000
