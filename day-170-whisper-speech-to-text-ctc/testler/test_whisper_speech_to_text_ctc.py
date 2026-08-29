"""
GÜN 170: OpenAI Whisper Speech-to-Text & CTC Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.log_mel_spektrogram_cikarici import LogMelSpektrogramCikarici
from src.whisper_modeli import WhisperModeli
from src.zaman_damgasi_ve_wer_degerlendirici import WhisperMetrikDegerlendirici
from src.gorsellestirici import WhisperGorsellestirici


def test_log_mel_spektrogram_cikarici_boyut():
    """16 kHz ses dalgasından 80 kanallı Log-Mel spektrogramı çıktığını test eder."""
    cikarici = LogMelSpektrogramCikarici(ornekleme_orani=16000, n_mels=80)
    ses = torch.randn(1, 16000)  # 1 saniyelik ses
    mel = cikarici.spektrogram_cikar(ses)

    assert mel.shape[0] == 1
    assert mel.shape[1] == 80
    assert mel.shape[2] == 100  # 16000 / 160 = 100 frame


def test_whisper_audio_encoder_ve_ctc():
    """Whisper ses kodlayıcısının Conv2 ve Transformer Encoder'dan geçtiğini test eder."""
    model = WhisperModeli(n_mels=80, d_model=64, num_encoder_layers=2, num_decoder_layers=2, vocab_size=200)
    mel = torch.randn(2, 80, 100)
    enc_out = model.audio_kodla(mel)

    # Conv2 ile zaman boyutu 2 kat küçülür: 100 -> 50
    assert enc_out.shape == (2, 50, 64)


def test_whisper_forward_cikis_logits():
    """Modelin hem Language Model (LM) hem de CTC logits ürettiğini test eder."""
    model = WhisperModeli(n_mels=80, d_model=64, num_encoder_layers=2, num_decoder_layers=2, vocab_size=200)
    mel = torch.randn(2, 80, 60)
    text_tokens = torch.randint(0, 200, (2, 10))
    lm_logits, ctc_logits = model(mel, text_tokens)

    assert lm_logits.shape == (2, 10, 200)
    assert ctc_logits.shape == (2, 30, 200)


def test_wer_hesaplama_tam_eslesme():
    """Birebir aynı metinlerde WER'in 0.0 olduğunu test eder."""
    g = "Yapay zeka modelleri çok dilli konuşmayı anlıyor"
    t = "Yapay zeka modelleri çok dilli konuşmayı anlıyor"
    assert WhisperMetrikDegerlendirici.wer_hesapla(g, t) == 0.0


def test_wer_hesaplama_farkli_metin():
    """Kelime hatalı metinde WER'in doğru hesaplandığını test eder."""
    g = "kedi masanın üstünde oturuyor"
    t = "kedi sandalyenin üstünde oturuyor"
    assert WhisperMetrikDegerlendirici.wer_hesapla(g, t) == 0.25


def test_cer_hesaplama():
    """Karakter hata oranının (CER) doğru hesaplandığını test eder."""
    g = "kitap"
    t = "katip"
    cer = WhisperMetrikDegerlendirici.cer_hesapla(g, t)
    assert cer > 0.0


def test_whisper_ornek_transkripsiyon_senaryolari():
    """Örnek Türkçe ve İngilizce senaryoların WER/CER oranlarını test eder."""
    rapor = WhisperModeli.ornek_transkripsiyon_senaryolarini_getir()
    assert len(rapor["senaryolar"]) == 2
    assert rapor["ortalama_wer"] == 0.0
    assert rapor["ortalama_cer"] == 0.0


def test_gorsellestirici_pano_uretme():
    """6 panelli Whisper teşhis panosunun PNG olarak kaydedildiğini test eder."""
    mel = torch.randn(1, 80, 100)
    rapor = WhisperModeli.ornek_transkripsiyon_senaryolarini_getir()
    gorsellestirici = WhisperGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_whisper_pano.png")
        gorsellestirici.pano_olustur(mel, rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
