"""
PyTest Birim Testleri - Day 190: Özel Triton FlashAttention-2.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.flash_attention_motoru import (
    PyTorchStandartAttention,
    FlashAttention2,
    FlashAttention2Function,
)
from src.hafiza_profilleyici import FlashAttentionBellekProfilleyici
from src.gorsellestirici import FlashAttentionGorsellestirici


@pytest.fixture
def test_qkv():
    """Test Q, K, V Tensörleri (B=1, H=4, S=128, D=64)."""
    torch.manual_seed(42)
    b, h, s, d = 1, 4, 128, 64
    q = torch.randn(b, h, s, d, requires_grad=True)
    k = torch.randn(b, h, s, d, requires_grad=True)
    v = torch.randn(b, h, s, d, requires_grad=True)
    return q, k, v


def test_flash_attention_non_causal_eslesme(test_qkv):
    """1. FlashAttention-2 non-causal çıktısı standart dikkat ile eşleşmelidir (< 1e-4)."""
    q, k, v = test_qkv
    std_out, _ = PyTorchStandartAttention()(q, k, v, causal=False)
    fa_out = FlashAttention2(causal=False)(q, k, v)
    assert torch.allclose(std_out, fa_out, atol=1e-4)


def test_flash_attention_causal_eslesme(test_qkv):
    """2. FlashAttention-2 nedensel (causal) çıktısı standart nedensel dikkat ile eşleşmelidir."""
    q, k, v = test_qkv
    std_out, _ = PyTorchStandartAttention()(q, k, v, causal=True)
    fa_out = FlashAttention2(causal=True)(q, k, v)
    assert torch.allclose(std_out, fa_out, atol=1e-4)


def test_flash_attention_farkli_blok_boyutlari(test_qkv):
    """3. Farklı blok boyutlarında (Br=32, Bc=32) FlashAttention doğru sonuç vermelidir."""
    q, k, v = test_qkv
    std_out, _ = PyTorchStandartAttention()(q, k, v, causal=False)
    fa_out = FlashAttention2Function.apply(q, k, v, False, 32, 32)
    assert torch.allclose(std_out, fa_out, atol=1e-4)


def test_flash_attention_autograd_gradyan_sekli(test_qkv):
    """4. FlashAttention-2 autograd geri geçişi gradyanları doğru şekillerde üretmelidir."""
    q, k, v = test_qkv
    fa_out = FlashAttention2(causal=False)(q, k, v)
    loss = fa_out.sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape


def test_flash_attention_uzun_dizi_stabilite():
    """5. Uzun dizilerde (S=256) sayısal kararlılık korunmalıdır."""
    torch.manual_seed(42)
    q = torch.randn(1, 2, 256, 32)
    k = torch.randn(1, 2, 256, 32)
    v = torch.randn(1, 2, 256, 32)

    std_out, _ = PyTorchStandartAttention()(q, k, v, causal=True)
    fa_out = FlashAttention2(causal=True)(q, k, v)
    assert torch.allclose(std_out, fa_out, atol=1e-4)


def test_bellek_profilleyici_o_n_tasarruf():
    """6. Bellek profilleyicisi O(N) FlashAttention'ın dramatik VRAM tasarrufunu doğrulamalıdır."""
    analiz = FlashAttentionBellekProfilleyici.baglam_uzunlugu_vram_analizi(
        batch_size=1, num_heads=32, head_dim=128, seq_len=8192
    )
    assert analiz["tasarruf_orani"] > 10.0
    assert analiz["flash_vram_mb"] < analiz["standart_vram_mb"]


def test_baglam_tarama_raporu():
    """7. Bağlam tarama raporu 1k'dan 128k'ya 5 seviyeyi içermeli ve 128k'da OOM tespit etmelidir."""
    rapor = FlashAttentionBellekProfilleyici.baglam_tarama_raporu()
    assert len(rapor) == 5
    son_eleman = rapor[-1]
    assert son_eleman["context_length"] == 131072
    assert "OOM" in son_eleman["standart_oom_durumu"]


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. FlashAttentionGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_flash_attention_paneli.png")
    katman_analizi = FlashAttentionBellekProfilleyici.baglam_uzunlugu_vram_analizi(seq_len=2048)
    baglam_raporu = FlashAttentionBellekProfilleyici.baglam_tarama_raporu()

    FlashAttentionGorsellestirici.teshis_paneli_olustur(
        katman_analizi=katman_analizi,
        baglam_raporu=baglam_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
