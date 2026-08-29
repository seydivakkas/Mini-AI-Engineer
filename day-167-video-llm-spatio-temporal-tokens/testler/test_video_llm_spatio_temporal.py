"""
GÜN 167: Video LLM Spatio-Temporal Token Modelleme Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.zamansal_ornekleyici import ZamansalKareOrnekleyici
from src.spatio_temporal_attention import SpatioTemporalAttention
from src.video_llava_modeli import VideoLLaVAModeli
from src.gorsellestirici import VideoLLMGorsellestirici


def test_uniform_zamansal_ornekleme():
    """60 kareden tam 8 adet eşit aralıklı kare seçildiğini test eder."""
    indeksler = ZamansalKareOrnekleyici.duzenli_ornekle(toplam_kare=60, ornek_sayisi=8)
    assert len(indeksler) == 8
    assert indeksler[0] == 0
    assert indeksler[-1] == 59


def test_adaptive_zamansal_ornekleme():
    """Harekete duyarlı adaptif kare seçiminin doğru indeksleri bulduğunu test eder."""
    skorlar = [0.1] * 20
    skorlar[5] = 0.9
    skorlar[12] = 0.85
    secilen = ZamansalKareOrnekleyici.uyarlamali_ornekle(skorlar, ornek_sayisi=4)
    assert 5 in secilen
    assert 12 in secilen


def test_spatio_temporal_attention_tensor_boyutlari():
    """3D Spatio-Temporal Attention'ın girdi ve çıktı tensor boyutlarını koruduğunu test eder."""
    st_attn = SpatioTemporalAttention(d_model=64, num_heads=2)
    x = torch.randn(2, 4, 16, 64)  # [Batch=2, T=4, N=16, D=64]
    out = st_attn(x)
    assert out.shape == (2, 4, 16, 64)


def test_video_llava_forward_ciktisi():
    """VideoLLaVA modelinin video tensorunu LLM token uzayına doğru yansıttığını test eder."""
    model = VideoLLaVAModeli(kare_sayisi=4, kare_basina_token=8, viz_dim=64, llm_dim=128)
    video = torch.randn(2, 4, 8, 64)  # [B=2, T=4, N=8, D=64]
    llm_tokens = model(video)
    # Çıktı boyutu: [B, T*N, llm_dim] = [2, 32, 128]
    assert llm_tokens.shape == (2, 32, 128)


def test_video_llava_ornek_qa_senaryolari():
    """Video-QA senaryo raporunun tam doğrulukla sonuçlandığını test eder."""
    rapor = VideoLLaVAModeli.ornek_video_qa_senaryolarini_degerlendir()
    assert len(rapor["senaryolar"]) == 2
    assert rapor["ortalama_video_qa_skoru"] == 1.0


def test_zamansal_ornekleme_sinir_durumu_kucuk_video():
    """Toplam kare sayısı örneklem sayısından az olduğunda tüm kareleri döndürdüğünü test eder."""
    indeksler = ZamansalKareOrnekleyici.duzenli_ornekle(toplam_kare=4, ornek_sayisi=8)
    assert indeksler == [0, 1, 2, 3]


def test_spatio_temporal_attention_gradyan_akisi():
    """Modelin geriye yayılımda (backward pass) gradyanları düzgün hesapladığını test eder."""
    st_attn = SpatioTemporalAttention(d_model=32, num_heads=2)
    x = torch.randn(1, 2, 4, 32, requires_grad=True)
    out = st_attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_gorsellestirici_pano_uretme():
    """6 panelli Video LLM teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = VideoLLaVAModeli.ornek_video_qa_senaryolarini_degerlendir()
    gorsellestirici = VideoLLMGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_video_llm_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
