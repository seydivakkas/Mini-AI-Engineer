"""
Distributed Data Parallel (DDP) Test Paketi (Day 181 - FAZ 10).
8 adet kapsamlı PyTest birim testi.
"""

import sys
import os
import tempfile
import pytest
import torch
import torch.nn as nn

# Proje dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ddp_iletisim_motoru import RingAllReduceSimulasyonu, GradyanPaketleyici
from src.dagitik_egitim_dongusu import DDPVeriOrnekleyici, DDPModelSarmalayici
from src.gorsellestirici import DDPGorsellestirici


def test_ring_all_reduce_matematiksel_dogruluk():
    """1. Ring All-Reduce ortalama hesabının doğruluğu ve tolerans testi."""
    num_ranks = 4
    dim = 500
    torch.manual_seed(42)

    rank_tensors = [torch.randn(dim) for _ in range(num_ranks)]
    expected_avg = sum(rank_tensors) / num_ranks

    engine = RingAllReduceSimulasyonu(num_ranks=num_ranks)
    synced, stats = engine.all_reduce(rank_tensors)

    for r in range(num_ranks):
        assert torch.allclose(synced[r], expected_avg, atol=1e-5), f"Rank {r} ortalama sonucu hatalı."

    assert stats["scatter_reduce_adim"] == 3
    assert stats["all_gather_adim"] == 3
    assert stats["toplam_ring_adimi"] == 6


def test_ring_all_reduce_farkli_rank_sayilari():
    """2. 2, 3, 5 ve 8 rank ile tek/çift boyutlu tensörlerde Ring All-Reduce testi."""
    for num_ranks in [2, 3, 5, 8]:
        dim = 127  # Asal sayı (tam bölünmez, padding zorunlu)
        tensors = [torch.ones(dim) * (r + 1.0) for r in range(num_ranks)]
        expected_avg = torch.ones(dim) * ((num_ranks + 1) / 2.0)

        engine = RingAllReduceSimulasyonu(num_ranks=num_ranks)
        synced, _ = engine.all_reduce(tensors)

        assert torch.allclose(synced[0], expected_avg, atol=1e-5)


def test_gradyan_paketleyici_bucket_kapasitesi():
    """3. Gradient Bucketing 25 MB havuzlama kapasitesi ve gruplama testi."""
    model = nn.Sequential(
        nn.Linear(50, 100),
        nn.Linear(100, 100),
        nn.Linear(100, 10),
    )
    params = list(model.parameters())

    # Küçük bucket kapasitesi (ör. 0.001 MB = 1 KB) vererek çoklu bucket oluşmasını sağla
    paketleyici = GradyanPaketleyici(bucket_cap_mb=0.005)
    buckets = paketleyici.parametreleri_paketle(params)

    assert len(buckets) >= 2, "Küçük kapasitede birden fazla bucket oluşmalıdır."

    # Toplam parametre sayısının korunduğunu doğrula
    total_in_buckets = sum(sum(p.numel() for p in b) for b in buckets)
    total_original = sum(p.numel() for p in params)
    assert total_in_buckets == total_original


def test_gradyan_paketleyici_ters_sira():
    """4. Gradient Bucketing'in geri geçiş için parametreleri tersten paketlemesi testi."""
    p1 = nn.Parameter(torch.zeros(10))
    p2 = nn.Parameter(torch.zeros(20))
    p3 = nn.Parameter(torch.zeros(30))

    paketleyici = GradyanPaketleyici(bucket_cap_mb=100.0)
    buckets = paketleyici.parametreleri_paketle([p1, p2, p3])

    # Tek bucket içinde ilk parametre p3 (son katman) olmalıdır
    assert buckets[0][0] is p3
    assert buckets[0][-1] is p1


def test_distributed_sampler_cakismasiz_indeksler():
    """5. DistributedSampler'ın rank'ler arasında çakışmasız indeks üretmesi testi."""
    dataset_size = 100
    num_ranks = 4

    all_indices = []
    for r in range(num_ranks):
        sampler = DDPVeriOrnekleyici(dataset_boyutu=dataset_size, num_replicas=num_ranks, rank=r, shuffle=False)
        indices = sampler.get_indices()
        assert len(indices) == 25, "Her rank tam 25 örnek almalıdır."
        all_indices.extend(indices)

    # İndekslerin benzersiz olduğunu ve [0..99] kümesini kapsadığını doğrula
    assert len(set(all_indices)) == dataset_size
    assert set(all_indices) == set(range(dataset_size))


def test_ddp_model_sarmalayici_agirlik_senkronizasyonu():
    """6. DDP eğitim adımı sonrasında tüm rank ağırlıklarının bitwise senkron kalması testi."""
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    ddp = DDPModelSarmalayici(model, num_ranks=3, lr=0.05)
    criterion = nn.MSELoss()

    rank_x = [torch.randn(4, 8) for _ in range(3)]
    rank_y = [torch.randn(4, 2) for _ in range(3)]

    # 3 eğitim adımı işlet
    for _ in range(3):
        res = ddp.egitim_adimi(rank_x, rank_y, criterion)

    assert res["senkronizasyon_basarili"] is True
    assert res["agirlik_senkron_farki"] < 1e-6, "Tüm rank modelleri birebir aynı ağırlıklara sahip olmalıdır."


def test_ddp_olceklenme_raporu_tutarliligi():
    """7. Çoklu GPU DDP ölçeklenme hız ve verimlilik raporu doğrulaması."""
    rapor = DDPModelSarmalayici.ornek_ddp_olceklenme_raporu()

    assert "karsilastirma" in rapor
    items = rapor["karsilastirma"]
    assert len(items) == 5

    # GPU sayısı arttıkça throughput kesinlikle artmalıdır
    hizlar = [item["hiz_imgs_per_sec"] for item in items]
    assert hizlar == sorted(hizlar), "Hızlar GPU sayısıyla monoton artmalıdır."


def test_gorsellestirme_cikti_dosyasi():
    """8. 6 panelli DDP teşhis panosunun kaydedilmesi testi."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kayit_yolu = os.path.join(tmpdir, "test_ddp.png")
        olcek = DDPModelSarmalayici.ornek_ddp_olceklenme_raporu()

        gorsellestirici = DDPGorsellestirici(dpi=100)
        gorsellestirici.pano_olustur(olcek_raporu=olcek, kayit_yolu=kayit_yolu)

        assert os.path.exists(kayit_yolu), "Görselleştirme dosyası kaydedilmiş olmalıdır."
        assert os.path.getsize(kayit_yolu) > 1000, "Dosya boyutu geçerli olmalıdır."
