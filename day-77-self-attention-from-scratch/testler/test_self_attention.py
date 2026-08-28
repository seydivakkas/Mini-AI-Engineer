"""
Sıfırdan Scaled Dot-Product & Multi-Head Self-Attention Birim Testleri
----------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import torch
import numpy as np

from src.scaled_dot_product import OlcekliNoktaCarpimDikkat
from src.multi_head_attention import CokKafaliOzDikkat
from src.dikkat_analizcisi import DikkatAnalizcisi


def test_scaled_dot_product_boyut_ve_olasilik():
    torch.manual_seed(42)
    b, h, n, d_k = 2, 4, 8, 16
    q = torch.randn(b, h, n, d_k)
    k = torch.randn(b, h, n, d_k)
    v = torch.randn(b, h, n, d_k)

    core = OlcekliNoktaCarpimDikkat(dropout_orani=0.0)
    cikti, agirliklar = core(q, k, v)

    assert cikti.shape == (b, h, n, d_k)
    assert agirliklar.shape == (b, h, n, n)
    
    # Satır toplamlarının 1.0 olduğunu doğrula
    satir_toplami = agirliklar.sum(dim=-1)
    assert torch.allclose(satir_toplami, torch.ones_like(satir_toplami), atol=1e-5)


def test_scaled_dot_product_causal_mask():
    torch.manual_seed(42)
    b, h, n, d_k = 1, 1, 4, 8
    q = torch.randn(b, h, n, d_k)
    k = torch.randn(b, h, n, d_k)
    v = torch.randn(b, h, n, d_k)

    # Üst üçgen maskesi (Geleceği görmeyi engelle)
    mask = torch.tril(torch.ones(n, n)).view(1, 1, n, n)

    core = OlcekliNoktaCarpimDikkat(dropout_orani=0.0)
    _, agirliklar = core(q, k, v, mask=mask)

    # Üst üçgen (j > i) ağırlıkları 0.0 olmalı
    for i in range(n):
        for j in range(i + 1, n):
            assert agirliklar[0, 0, i, j].item() < 1e-6, f"Maskelenmiş ({i},{j}) pozisyonu sıfır olmalıdır!"


def test_mhsa_ileri_gecis_boyutlari():
    torch.manual_seed(42)
    b, n, d_model = 3, 12, 64
    kafa_sayisi = 4
    
    mhsa = CokKafaliOzDikkat(model_boyutu=d_model, kafa_sayisi=kafa_sayisi)
    x = torch.randn(b, n, d_model)
    
    cikti, dikkat_haritasi = mhsa(x)

    assert cikti.shape == (b, n, d_model)
    assert dikkat_haritasi.shape == (b, kafa_sayisi, n, n)


def test_mhsa_kafa_sayisi_uyumsuzlugu():
    with pytest.raises(AssertionError):
        # 64, 5'e tam bölünmez
        _ = CokKafaliOzDikkat(model_boyutu=64, kafa_sayisi=5)


def test_mhsa_gradyan_akisi():
    torch.manual_seed(42)
    mhsa = CokKafaliOzDikkat(model_boyutu=32, kafa_sayisi=2)
    x = torch.randn(2, 8, 32, requires_grad=True)

    cikti, _ = mhsa(x)
    loss = cikti.sum()
    loss.backward()

    assert mhsa.w_q.weight.grad is not None
    assert mhsa.w_k.weight.grad is not None
    assert mhsa.w_v.weight.grad is not None
    assert mhsa.w_o.weight.grad is not None
    assert x.grad is not None
    assert x.grad.norm().item() > 0.0


def test_dikkat_analizcisi_entropi_ve_mesafe():
    torch.manual_seed(42)
    b, h, n = 2, 4, 10
    # Düzgün dağılım oluştur
    dikkat = torch.ones(b, h, n, n) / n

    analizci = DikkatAnalizcisi()
    entropiler = analizci.hesapla_dikkat_entropisi(dikkat)
    mesafeler = analizci.hesapla_dikkat_mesafesi(dikkat)
    cesitlilik = analizci.hesapla_baslar_arasi_cesitlilik(dikkat)

    assert len(entropiler) == h
    assert len(mesafeler) == h
    assert np.all(entropiler > 0.0)
    assert np.all(mesafeler > 0.0)
    assert isinstance(cesitlilik, float)


def test_dikkat_analizcisi_olcek_etkisi():
    analizci = DikkatAnalizcisi()
    res = analizci.olcek_etkisi_analizi(d_k=64, seq_len=16)

    # 1/sqrt(d_k) ölçeklemesi entropiyi artırmalı (dağılımı aşırı sivriltmekten korumalı)
    assert res["olcekli_entropi"] > res["olceksiz_entropi"]
    assert res["olcekli_skor_std"] < res["olceksiz_skor_std"]


def test_mhsa_cross_attention_destegi():
    torch.manual_seed(42)
    b, n_q, n_kv, d = 2, 6, 10, 32
    q = torch.randn(b, n_q, d)
    k = torch.randn(b, n_kv, d)
    v = torch.randn(b, n_kv, d)

    mhsa = CokKafaliOzDikkat(model_boyutu=d, kafa_sayisi=4)
    cikti, dikkat = mhsa(q, k, v)

    assert cikti.shape == (b, n_q, d)
    assert dikkat.shape == (b, 4, n_q, n_kv)
