"""
PyTest Birim Testleri - Day 187: OpenAI Triton GPU Kernel Programlama.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.triton_temel_motoru import (
    TritonBlokSimulasyonu,
    VektorToplamaKernel,
    FusedLineerKombinasyonKernel,
)
from src.bellek_esleme_profilleyici import TritonBellekProfilleyici
from src.gorsellestirici import TritonGorsellestirici


def test_triton_grid_ve_ofset_hesaplama():
    """1. hesapla_grid_ve_ofset() grid boyutunu doğru hesaplamalı, geçersiz blok boyutlarında hata vermelidir."""
    blocks, b_sz = TritonBlokSimulasyonu.hesapla_grid_ve_ofset(n_eleman=2500, block_size=1024)
    assert blocks == 3
    assert b_sz == 1024

    with pytest.raises(AssertionError):
        TritonBlokSimulasyonu.hesapla_grid_ve_ofset(n_eleman=1000, block_size=500)  # 2'nin kuvveti değil


def test_vektor_toplama_tam_bolunen_dizi():
    """2. Vektör toplama tam bölünen boyutta (N=4096) PyTorch ile birebir eşleşmelidir."""
    x = torch.randn(4096)
    y = torch.randn(4096)
    z = VektorToplamaKernel.calistir(x, y, block_size=1024)
    assert torch.allclose(z, x + y, atol=1e-6)


def test_vektor_toplama_asal_sayi_maskeleme():
    """3. Sınır maskeleme (offsets < N) asal sayılı boyutta (N=54321) sıfır taşma ile çalışmalıdır."""
    x = torch.randn(54321)
    y = torch.randn(54321)
    z = VektorToplamaKernel.calistir(x, y, block_size=1024)
    assert z.shape == (54321,)
    assert torch.allclose(z, x + y, atol=1e-6)


def test_fused_lineer_kombinasyon_dogruluk():
    """4. FusedLineerKombinasyonKernel Y = alpha*X1 + beta*X2 + gamma işlemini doğru hesaplamalıdır."""
    x1 = torch.randn(10000)
    x2 = torch.randn(10000)
    alpha, beta, gamma = 2.5, -1.8, 0.45

    out = FusedLineerKombinasyonKernel.calistir(x1, x2, alpha=alpha, beta=beta, gamma=gamma, block_size=1024)
    expected = alpha * x1 + beta * x2 + gamma
    assert torch.allclose(out, expected, atol=1e-5)


def test_tl_load_ve_store_sinir_guvenligi():
    """5. tl_load ve tl_store sınır ötesi indekslerde güvenli maskeleme yapmalıdır."""
    t = torch.tensor([10.0, 20.0, 30.0])
    offsets = torch.tensor([0, 1, 2, 3, 4])
    mask = offsets < 3

    val = TritonBlokSimulasyonu.tl_load(t, offsets, mask=mask, other=-1.0)
    assert torch.allclose(val[:3], t)
    assert val[3].item() == -1.0
    assert val[4].item() == -1.0


def test_bellek_profilleyici_tasarruf_orani():
    """6. Bellek profilleyici Triton'un HBM trafiğini tam 3.0x azalttığını (%66.7 kazanç) doğrulamalıdır."""
    analiz = TritonBellekProfilleyici.lineer_kombinasyon_bellek_analizi(eleman_sayisi=1_000_000)
    assert analiz["tasarruf_orani"] == pytest.approx(3.0, abs=1e-2)
    assert analiz["triton_ara_bellek_mb"] == 0.0


def test_blok_boyutu_tarama_raporu():
    """7. blok_boyutu_tarama_raporu 5 farklı BLOCK_SIZE için doğru grid analizini dönmelidir."""
    rapor = TritonBellekProfilleyici.blok_boyutu_tarama_raporu(eleman_sayisi=10_000_000)
    assert len(rapor) == 5
    b_sizes = [r["block_size"] for r in rapor]
    assert b_sizes == [128, 256, 512, 1024, 2048]


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. TritonGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_triton_paneli.png")
    analiz = TritonBellekProfilleyici.lineer_kombinasyon_bellek_analizi(eleman_sayisi=1_000_000)
    blok_raporu = TritonBellekProfilleyici.blok_boyutu_tarama_raporu(eleman_sayisi=1_000_000)

    TritonGorsellestirici.teshis_paneli_olustur(
        bellek_analizi=analiz,
        blok_raporu=blok_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
