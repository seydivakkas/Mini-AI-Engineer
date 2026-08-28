"""
Temsil Kalitesi Değerlendirme Paketi Birim Testleri
--------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.temsil_cikarici import TemsilCikarici
from src.linear_probe import DogrusalYoklayici, LinearProbeProtokolu
from src.knn_degerlendirici import KNNDegerlendirici
from src.benchmark_suite import TemsilDegerlendirmePaketi


class SahteModel(nn.Module):
    def __init__(self, giris: int = 10, cikti: int = 32):
        super().__init__()
        self.net = nn.Linear(giris, cikti)

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def sahte_veri():
    torch.manual_seed(42)
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 4, (100,))
    x_val = torch.randn(40, 10)
    y_val = torch.randint(0, 4, (40,))
    return x_train, y_train, x_val, y_val


def test_temsil_cikarici_boyut_ve_normalizasyon(sahte_veri):
    x_train, y_train, _, _ = sahte_veri
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32)
    model = SahteModel(10, 32)
    cikarici = TemsilCikarici(model, cihaz="cpu")
    
    temsiller, etiketler = cikarici.cikar(loader, normalize_et=True)
    
    assert temsiller.shape == (100, 32)
    assert etiketler.shape == (100,)
    # Norm testi
    normlar = torch.norm(temsiller, p=2, dim=1)
    assert torch.allclose(normlar, torch.ones_like(normlar), atol=1e-4)


def test_temsil_cikarici_omurga_dondurma():
    model = SahteModel(10, 32)
    cikarici = TemsilCikarici(model, cihaz="cpu")
    for param in model.parameters():
        assert param.requires_grad is False


def test_linear_probe_egitim_dongusu():
    torch.manual_seed(42)
    h_train = torch.randn(80, 32)
    y_train = torch.randint(0, 4, (80,))
    h_val = torch.randn(20, 32)
    y_val = torch.randint(0, 4, (20,))

    protokol = LinearProbeProtokolu(temsil_boyutu=32, sinif_sayisi=4, cihaz="cpu")
    sonuc = protokol.egit_ve_degerlendir(h_train, y_train, h_val, y_val, etiket_orani=1.0, epoch_sayisi=5)

    assert "dogruluk_yuzdesi" in sonuc
    assert "dogrulama_kaybi" in sonuc
    assert 0.0 <= sonuc["dogruluk_yuzdesi"] <= 100.0


def test_linear_probe_fewshot_ornekleme():
    torch.manual_seed(42)
    h_train = torch.randn(100, 32)
    y_train = torch.randint(0, 4, (100,))
    h_val = torch.randn(20, 32)
    y_val = torch.randint(0, 4, (20,))

    protokol = LinearProbeProtokolu(temsil_boyutu=32, sinif_sayisi=4, cihaz="cpu")
    sonuc = protokol.egit_ve_degerlendir(h_train, y_train, h_val, y_val, etiket_orani=0.10, epoch_sayisi=5)

    assert sonuc["kullanilan_ornek_sayisi"] == 10
    assert sonuc["etiket_orani"] == 0.10


def test_knn_degerlendirici_k_degerleri():
    torch.manual_seed(42)
    h_train = torch.randn(60, 32)
    y_train = torch.randint(0, 3, (60,))
    h_val = torch.randn(15, 32)
    y_val = torch.randint(0, 3, (15,))

    knn = KNNDegerlendirici(sicaklik=0.07)
    sonuclar = knn.degerlendir(h_train, y_train, h_val, y_val, k_degerleri=[1, 5, 10], sinif_sayisi=3)

    assert "knn_k_1" in sonuclar
    assert "knn_k_5" in sonuclar
    assert "knn_k_10" in sonuclar
    for k_key, val in sonuclar.items():
        assert 0.0 <= val <= 100.0


def test_knn_degerlendirici_sicaklik_etkisi():
    # Çok küçük ve büyük sıcaklık değerlerinde taşma olmamalı
    torch.manual_seed(42)
    h_train = torch.randn(50, 16)
    y_train = torch.randint(0, 2, (50,))
    h_val = torch.randn(10, 16)
    y_val = torch.randint(0, 2, (10,))

    knn_dusuk = KNNDegerlendirici(sicaklik=0.01)
    knn_yuksek = KNNDegerlendirici(sicaklik=1.0)

    res1 = knn_dusuk.degerlendir(h_train, y_train, h_val, y_val, k_degerleri=[1, 5], sinif_sayisi=2)
    res2 = knn_yuksek.degerlendir(h_train, y_train, h_val, y_val, k_degerleri=[1, 5], sinif_sayisi=2)

    assert isinstance(res1["knn_k_1"], float)
    assert isinstance(res2["knn_k_1"], float)


def test_geometrik_metrikler_hesaplama():
    torch.manual_seed(42)
    # 2 belirgin küme
    c1 = torch.randn(25, 16) + 3.0
    c2 = torch.randn(25, 16) - 3.0
    h = torch.cat([c1, c2], dim=0)
    y = torch.tensor([0] * 25 + [1] * 25)

    suite = TemsilDegerlendirmePaketi(temsil_boyutu=16, sinif_sayisi=2, cihaz="cpu")
    geo = suite.hesapla_geometrik_metrikler(h, y)

    assert "silhouette_skoru" in geo
    assert "efektif_boyut" in geo
    assert "izotropi_indeksi" in geo
    assert "ayrisma_marjini" in geo
    assert geo["silhouette_skoru"] > 0.0 # Belirgin kümeler pozitif silhouette üretmeli


def test_benchmark_suite_tam_entegrasyon():
    torch.manual_seed(42)
    h_train = torch.randn(80, 16)
    y_train = torch.randint(0, 3, (80,))
    h_val = torch.randn(20, 16)
    y_val = torch.randint(0, 3, (20,))

    suite = TemsilDegerlendirmePaketi(temsil_boyutu=16, sinif_sayisi=3, cihaz="cpu")
    sonuclar = suite.calistir_kapsamli_benchmark(h_train, y_train, h_val, y_val)

    zorunlu_metrikler = [
        "linear_probe_100",
        "linear_probe_10",
        "linear_probe_fewshot",
        "knn_k_1",
        "knn_k_5",
        "silhouette_skoru",
        "izotropi_indeksi",
        "efektif_boyut",
        "ayrisma_marjini"
    ]
    for m in zorunlu_metrikler:
        assert m in sonuclar, f"{m} metriği sonuç sözlüğünde bulunamadı."
