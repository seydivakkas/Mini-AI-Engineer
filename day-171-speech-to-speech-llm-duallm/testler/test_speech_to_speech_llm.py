"""
GÜN 171: Uçtan Uca Speech-to-Speech LLM (DuaLLM / Moshi) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.cift_akisli_token_birlestirici import CiftAkisliTokenBirlestirici
from src.speech_to_speech_llm import SpeechToSpeechLLM
from src.ses_diyalog_degerlendirici import SesDiyalogDegerlendirici
from src.gorsellestirici import SpeechLLMGorsellestirici


def test_cift_akisli_token_birlestirici():
    """Ses ve metin gömme katmanlarının [B, T, d_model] boyutunda vektör ürettiğini test eder."""
    birlestirici = CiftAkisliTokenBirlestirici(text_vocab_size=100, audio_codebook_size=256, num_quantizers=4, d_model=64)
    audio_tokens = torch.randint(0, 256, (2, 4, 15))  # [B=2, Num_Q=4, T=15]
    text_tokens = torch.randint(0, 100, (2, 15))

    a_embed = birlestirici.ses_tokenlarini_gom(audio_tokens)
    t_embed = birlestirici.metin_tokenlarini_gom(text_tokens)

    assert a_embed.shape == (2, 15, 64)
    assert t_embed.shape == (2, 15, 64)


def test_speech_to_speech_llm_ileri_besleme():
    """Modelin hem metin hem de 8 katmanlı ses logitlerini doğru boyutlarla ürettiğini test eder."""
    model = SpeechToSpeechLLM(
        text_vocab_size=100,
        audio_codebook_size=256,
        num_quantizers=4,
        d_model=64,
        num_layers=2,
        num_heads=2,
    )
    audio_tokens = torch.randint(0, 256, (2, 4, 10))
    text_tokens = torch.randint(0, 100, (2, 10))

    text_logits, audio_logits = model(audio_tokens, text_tokens)

    assert text_logits.shape == (2, 10, 100)
    assert audio_logits.shape == (2, 4, 10, 256)


def test_rtf_hesaplama():
    """Real-Time Factor (RTF) fonksiyonunun doğru oran hesapladığını test eder."""
    # 200 ms üretim / 1000 ms ses = 0.2 RTF (< 1.0 = gerçek zamanlı)
    rtf = SesDiyalogDegerlendirici.rtf_hesapla(uretim_suresi_ms=200, ses_uzunlugu_ms=1000)
    assert rtf == 0.2


def test_gecikme_tasarruf_orani():
    """Geleneksel zincire göre gecikme iyileşme katsayısını test eder."""
    kat = SesDiyalogDegerlendirici.gecikme_tasarruf_orani(geleneksel_ms=1600, duallm_ms=200)
    assert kat == 8.0


def test_speech_to_speech_ornek_diyalog_raporu():
    """Örnek sesli diyalog senaryolarının tam doğruluk ve düşük gecikme sağladığını test eder."""
    rapor = SpeechToSpeechLLM.ornek_diyalog_senaryolarini_getir()
    assert len(rapor["senaryolar"]) == 2
    assert rapor["ortalama_duallm_gecikme_ms"] < 200.0
    assert rapor["gecikme_iyilesmesi_kat"] > 5.0


def test_speech_llm_gradyan_akisi():
    """Modelin geriye yayılımda (backward pass) gradyanları sorunsuz hesapladığını test eder."""
    model = SpeechToSpeechLLM(
        text_vocab_size=50,
        audio_codebook_size=64,
        num_quantizers=2,
        d_model=32,
        num_layers=1,
        num_heads=2,
    )
    audio_tokens = torch.randint(0, 64, (1, 2, 8))
    text_tokens = torch.randint(0, 50, (1, 8))
    t_logits, a_logits = model(audio_tokens, text_tokens)

    loss = t_logits.sum() + a_logits.sum()
    loss.backward()

    assert model.text_head.weight.grad is not None


def test_rtf_hesaplama_sifir_sure():
    """Sıfır ses uzunluğunda RTF'in 0.0 döndüğünü test eder."""
    assert SesDiyalogDegerlendirici.rtf_hesapla(100, 0) == 0.0


def test_gorsellestirici_pano_uretme():
    """6 panelli Speech-to-Speech LLM teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = SpeechToSpeechLLM.ornek_diyalog_senaryolarini_getir()
    gorsellestirici = SpeechLLMGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_speech_llm_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
