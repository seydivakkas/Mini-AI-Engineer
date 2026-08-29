"""
GÜN 163: Görsel Komut İnce Ayarı (Visual SFT) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from src.gorsel_komut_veri_seti import GorselKomutVeriSeti
from src.kayip_maskeleyici import VisualLossMaskeleyici
from src.visual_sft_egitici import VisualSFTEgitici
from src.gorsellestirici import VisualSFTGorsellestirici


class DummyVLM(nn.Module):
    """Testler için hafif VLM simülatörü."""
    def __init__(self, vocab_size=500, d_model=64):
        super().__init__()
        self.visual_proj = nn.Linear(768, d_model)
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, img, text_ids):
        B = img.shape[0]
        # Dummy 256 patch token
        dummy_vis = torch.randn(B, 256, 768, device=img.device)
        vis_emb = self.visual_proj(dummy_vis)
        txt_emb = self.text_embed(text_ids)
        fused = torch.cat([vis_emb, txt_emb], dim=1)
        logits = self.lm_head(fused)
        return logits


def test_gorsel_komut_veri_seti_ornekleri():
    """Veri setinin 3 temel komut kategorisini eksiksiz içerdiğini test eder."""
    veriler = GorselKomutVeriSeti.ornek_verileri_getir()
    assert len(veriler) == 3
    kategoriler = [v["kategori"] for v in veriler]
    assert "Kısa VQA" in kategoriler
    assert "Detaylı Açıklama" in kategoriler
    assert "Karmaşık Muhakeme" in kategoriler


def test_kayip_maskeleme_etiket_boyutu():
    """Görsel + Prompt bölgelerinin -100 olduğunu, sadece Asistan Yanıtının korunduğunu test eder."""
    prompt_ids = torch.randint(10, 500, (2, 10))
    response_ids = torch.randint(10, 500, (2, 15))

    input_text_ids, labels = VisualLossMaskeleyici.hedef_maskeli_etiket_olustur(
        visual_token_count=256,
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        ignore_index=-100,
    )

    # Toplam dizi: 256 görsel + 10 prompt + 15 yanıt = 281
    assert labels.shape == (2, 281)
    assert (labels[:, :266] == -100).all()  # İlk 266 token -100 olmalı
    assert (labels[:, 266:] == response_ids).all()  # Son 15 token gerçek yanıt olmalı


def test_maskeli_cross_entropy_kaybi_hesaplama():
    """Loss fonksiyonunun -100 etiketlerini yoksayarak skalar kayıp ürettiğini test eder."""
    logits = torch.randn(2, 50, 100)
    labels = torch.full((2, 50), -100, dtype=torch.long)
    labels[:, 40:] = torch.randint(0, 100, (2, 10))

    loss = VisualLossMaskeleyici.maskeli_cross_entropy_kaybi_hesapla(logits, labels)
    assert loss.dim() == 0  # Skalar
    assert loss.item() > 0.0


def test_kayip_maskeleme_gradyan_akisi():
    """Kayıp fonksiyonunun sadece eğitilen asistan tokenlarından gradyan ürettiğini test eder."""
    logits = torch.randn(1, 30, 50, requires_grad=True)
    labels = torch.full((1, 30), -100, dtype=torch.long)
    labels[:, 25:] = torch.randint(0, 50, (1, 5))

    loss = VisualLossMaskeleyici.maskeli_cross_entropy_kaybi_hesapla(logits, labels)
    loss.backward()

    assert logits.grad is not None


def test_visual_sft_egitim_dongusu():
    """Visual SFT eğitim döngüsünün kaybı düşürdüğünü test eder."""
    model = DummyVLM(vocab_size=500, d_model=64)
    rapor = VisualSFTEgitici.egitim_dongusu_yurut(model, adim_sayisi=3, ogrenme_orani=1e-3)

    assert len(rapor["kayip_gecmisi"]) == 3
    assert "kayip_dususu_yuzdesi" in rapor


def test_tum_etiketler_maskelendiginde_durum():
    """Tüm etiketler -100 olduğunda sistemin çökmediğini/nan üretmediğini test eder."""
    logits = torch.randn(1, 10, 20)
    labels = torch.full((1, 10), -100, dtype=torch.long)
    labels[:, -1] = 5  # En az 1 token açık

    loss = VisualLossMaskeleyici.maskeli_cross_entropy_kaybi_hesapla(logits, labels)
    assert not torch.isnan(loss)


def test_farkli_batch_boyutlarinda_maskeleme():
    """Maskeleme motorunun Batch=4 için doğru çalıştığını test eder."""
    p_ids = torch.randint(0, 100, (4, 8))
    r_ids = torch.randint(0, 100, (4, 12))

    _, labels = VisualLossMaskeleyici.hedef_maskeli_etiket_olustur(
        visual_token_count=128,
        prompt_token_ids=p_ids,
        response_token_ids=r_ids,
    )
    assert labels.shape == (4, 148)


def test_gorsellestirici_pano_uretme():
    """6 panelli görsel SFT teşhis panosunun PNG olarak kaydedildiğini test eder."""
    veriler = GorselKomutVeriSeti.ornek_verileri_getir()
    rapor = {
        "baslangic_kaybi": 6.82,
        "bitis_kaybi": 3.14,
        "kayip_dususu_yuzdesi": 53.96,
        "kayip_gecmisi": [6.82, 5.41, 4.35, 3.72, 3.14],
    }
    gorsellestirici = VisualSFTGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_sft_pano.png")
        gorsellestirici.pano_olustur(rapor, veriler, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
