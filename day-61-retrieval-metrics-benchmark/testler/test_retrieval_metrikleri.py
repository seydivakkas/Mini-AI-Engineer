"""
Day 61: Vektör ve Semantik Arama Değerlendirme (NDCG, MRR, MAP, Latency) Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.metrik_motoru import RetrievalMetrikMotoru
from src.arama_degerlendirici import AramaDegerlendirici
from src.gorsellestirici import RetrievalGorsellestirici


def test_precision_and_recall_at_k():
    """Precision@k ve Recall@k formüllerinin matematiksel doğruluğunu test eder."""
    retrieved = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = {2, 4, 6, 8, 12, 14}  # 6 toplam ilgili, ilk 5'te 2 adet (2,4), ilk 10'da 4 adet

    p5 = RetrievalMetrikMotoru.precision_at_k(retrieved, relevant, k=5)
    r5 = RetrievalMetrikMotoru.recall_at_k(retrieved, relevant, k=5)
    p10 = RetrievalMetrikMotoru.precision_at_k(retrieved, relevant, k=10)
    r10 = RetrievalMetrikMotoru.recall_at_k(retrieved, relevant, k=10)

    assert p5 == pytest.approx(2 / 5, abs=1e-5)
    assert r5 == pytest.approx(2 / 6, abs=1e-5)
    assert p10 == pytest.approx(4 / 10, abs=1e-5)
    assert r10 == pytest.approx(4 / 6, abs=1e-5)


def test_reciprocal_rank_calculation():
    """MRR (Reciprocal Rank) hesaplamasını test eder."""
    relevant = {42, 99}

    # 1. sırada bulundu -> RR = 1.0
    rr1 = RetrievalMetrikMotoru.reciprocal_rank([42, 1, 2], relevant)
    assert rr1 == 1.0

    # 2. sırada bulundu -> RR = 0.5
    rr2 = RetrievalMetrikMotoru.reciprocal_rank([1, 42, 2], relevant)
    assert rr2 == 0.5

    # 4. sırada bulundu -> RR = 0.25
    rr4 = RetrievalMetrikMotoru.reciprocal_rank([1, 2, 3, 99], relevant)
    assert rr4 == 0.25

    # Hiç bulunamadı -> RR = 0.0
    rr0 = RetrievalMetrikMotoru.reciprocal_rank([1, 2, 3], relevant)
    assert rr0 == 0.0


def test_average_precision_at_k():
    """Average Precision (AP@k) hesaplamasını test eder."""
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3}  # İlgililer 1. ve 3. sırada. Prec@1 = 1/1, Prec@3 = 2/3. AP = (1 + 2/3)/2 = 5/6

    ap = RetrievalMetrikMotoru.average_precision_at_k(retrieved, relevant, k=5)
    assert ap == pytest.approx(5 / 6, abs=1e-5)


def test_dcg_and_ndcg_at_k():
    """DCG ve NDCG@k dereceli ilgililik hesaplamalarını test eder."""
    true_relevances = [3.0, 2.0, 1.0, 0.0]

    # İdeal sıralama -> NDCG = 1.0
    ideal_order = [3.0, 2.0, 1.0, 0.0]
    ndcg_ideal = RetrievalMetrikMotoru.ndcg_at_k(ideal_order, true_relevances, k=4)
    assert ndcg_ideal == pytest.approx(1.0, abs=1e-5)

    # Kötü sıralama -> NDCG < 1.0
    bad_order = [0.0, 1.0, 2.0, 3.0]
    ndcg_bad = RetrievalMetrikMotoru.ndcg_at_k(bad_order, true_relevances, k=4)
    assert 0.0 < ndcg_bad < 1.0


def test_gecikme_profili_cikar():
    """Gecikme serisi istatistikleri ve QPS hesaplamasını test eder."""
    gecikmeler = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    profil = RetrievalMetrikMotoru.gecikme_profili_cikar(gecikmeler)

    assert profil["ortalama_ms"] == pytest.approx(5.5, abs=1e-3)
    assert profil["p50_ms"] == pytest.approx(5.5, abs=1e-3)
    assert profil["min_ms"] == 1.0
    assert profil["max_ms"] == 10.0
    assert profil["qps"] == pytest.approx(1000.0 / 5.5, abs=1e-2)


def test_arama_degerlendirici_karsilastirma():
    """AramaDegerlendirici'nin tüm arama stratejilerini kıyaslayıp geçerli metrikler ürettiğini test eder."""
    senaryolar = AramaDegerlendirici.sentetik_arama_senaryosu_uret(num_queries=10, katalog_boyutu=100)
    sonuclar = AramaDegerlendirici.calistir_karsilastirma(senaryolar)

    assert "Hybrid RRF (Dense + BM25)" in sonuclar
    assert "Dense Vector (HNSW)" in sonuclar
    assert "Lexical BM25 (Keyword)" in sonuclar
    assert "Approx IVF-Flat (Fast Baseline)" in sonuclar

    for strat, m in sonuclar.items():
        assert 0.0 <= m["ndcg@10"] <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0
        assert m["gecikme_istatistikleri"]["p50_ms"] > 0.0


def test_retrieval_gorsellestirici_panel(tmp_path):
    """6 panelli teşhis panosunun geçerli bir PNG dosyası ürettiğini test eder."""
    senaryolar = AramaDegerlendirici.sentetik_arama_senaryosu_uret(num_queries=5, katalog_boyutu=50)
    sonuclar = AramaDegerlendirici.calistir_karsilastirma(senaryolar)

    hedef = str(tmp_path / "test_retrieval_paneli.png")
    cikis = RetrievalGorsellestirici.panel_ciz(sonuclar, num_queries=5, hedef_path=hedef)

    assert os.path.exists(cikis)
    assert os.path.getsize(cikis) > 1000
