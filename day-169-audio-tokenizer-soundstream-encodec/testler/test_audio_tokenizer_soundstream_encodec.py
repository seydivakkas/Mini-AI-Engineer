"""
GÜN 169: Sinirsel Ses Sıkıştırma (EnCodec / SoundStream & RVQ) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.residual_vector_quantizer import ResidualVectorQuantizer, VektorKuantalayici
from src.encodec_modeli import NeuralAudioCodec
from src.ses_metrik_degerlendirici import SesMetrikDegerlendirici
from src.gorsellestirici import AudioTokenizerGorsellestirici


def test_vektor_kuantalayici_ileri_besleme():
    """Tekli VQ katmanının doğru şekilli kuantalanmış vektör ve indeksler ürettiğini test eder."""
    vq = VektorKuantalayici(codebook_size=256, dim=64)
    z = torch.randn(2, 20, 64)
    z_q, indices, loss = vq(z)

    assert z_q.shape == (2, 20, 64)
    assert indices.shape == (2, 20)
    assert loss.item() >= 0.0


def test_residual_vector_quantizer_kademeli():
    """RVQ modülünün N_q=4 katmanda tüm indeksleri [B, N_q, T] şeklinde ürettiğini test eder."""
    rvq = ResidualVectorQuantizer(num_quantizers=4, codebook_size=256, dim=64)
    z = torch.randn(2, 15, 64)
    z_q, all_indices, loss = rvq(z)

    assert z_q.shape == (2, 15, 64)
    assert all_indices.shape == (2, 4, 15)
    assert loss.item() >= 0.0


def test_encodec_forward_rekonstruksiyon_boyutu():
    """NeuralAudioCodec modelinin ham ses dalgasını aynı uzunlukta yeniden yapılandırdığını test eder."""
    codec = NeuralAudioCodec(in_channels=1, hidden_dim=64, num_quantizers=4, codebook_size=256)
    audio = torch.randn(1, 1, 2400)  # 0.1 saniyelik ses
    recon_wave, tokens, loss = codec(audio)

    assert recon_wave.shape == audio.shape
    assert tokens.shape[1] == 4  # Num_Q = 4


def test_snr_metrik_hesaplama():
    """Birebir aynı sinyalde SNR'ın çok yüksek (> 50 dB) çıktığını test eder."""
    ref = torch.sin(torch.linspace(0, 100, 1000))
    est = ref.clone()
    snr = SesMetrikDegerlendirici.snr_hesapla(ref, est)
    assert snr > 50.0


def test_bitrate_hesaplama():
    """75 fps, 8 kuantalayıcı ve 10-bit kod defteri ile bitrate'in 6.0 kbps olduğunu test eder."""
    bitrate = SesMetrikDegerlendirici.bitrate_hesapla(kare_orani_hz=75, num_quantizers=8, codebook_bits=10)
    assert bitrate == 6.0


def test_codebook_perplexity_hesaplama():
    """Token dağılımı perplexity'sinin pozitif olduğunu test eder."""
    tokens = torch.randint(0, 1024, (2, 8, 50))
    perp = SesMetrikDegerlendirici.codebook_perplexity_hesapla(tokens, codebook_size=1024)
    assert perp > 1.0


def test_encodec_gradyan_akisi():
    """Straight-Through Estimator sayesinde modelin uçtan uca gradyan aldığını test eder."""
    codec = NeuralAudioCodec(in_channels=1, hidden_dim=32, num_quantizers=2, codebook_size=128)
    audio = torch.randn(1, 1, 1200, requires_grad=True)
    recon, tokens, commit_loss = codec(audio)
    loss = torch.nn.functional.mse_loss(recon, audio) + commit_loss
    loss.backward()

    assert audio.grad is not None


def test_gorsellestirici_pano_uretme():
    """6 panelli Audio Tokenizer teşhis panosunun PNG olarak kaydedildiğini test eder."""
    audio = torch.sin(torch.linspace(0, 50, 2400)).unsqueeze(0).unsqueeze(0)
    recon = audio + torch.randn_like(audio) * 0.05
    metrikler = {
        "snr_db": 24.5,
        "bitrate_kbps": 6.0,
        "perplexity": 820.5,
    }
    gorsellestirici = AudioTokenizerGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_audio_pano.png")
        gorsellestirici.pano_olustur(audio, recon, metrikler, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
