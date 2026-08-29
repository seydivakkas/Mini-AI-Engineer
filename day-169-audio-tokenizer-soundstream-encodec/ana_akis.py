"""
Day 169: Sinirsel Ses Sıkıştırma (EnCodec / SoundStream & RVQ) Ana Akışı.
24 kHz Sürekli Ses Dalgasını 8 Katmanlı Ayrık Tokenlara Bölme ve Yeniden Yapılandırma.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.encodec_modeli import NeuralAudioCodec
from src.ses_metrik_degerlendirici import SesMetrikDegerlendirici
from src.gorsellestirici import AudioTokenizerGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 169 (FAZ 9): NEURAL AUDIO TOKENIZATION: ENCODEC / SOUNDSTREAM & RESIDUAL VECTOR QUANTIZATION (RVQ)")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. 24 kHz SENTETİK SES SİNYALİ ÜRETİMİ (1 Saniye = 24.000 Örneklem)
    # -------------------------------------------------------------
    print("\n[1/3] 24 kHz Sentetik Ses Sinyali (Harmonik Konuşma Benzetimi) Oluşturuluyor...")
    t = torch.linspace(0, 1.0, 24000)
    # 440 Hz (La) + 880 Hz harmonik + genlik modülasyonu
    audio_wave = (
        0.6 * torch.sin(2 * torch.pi * 440 * t)
        + 0.3 * torch.sin(2 * torch.pi * 880 * t)
        + 0.1 * torch.sin(2 * torch.pi * 1320 * t)
    ) * torch.exp(-t * 2.0)
    audio_wave = audio_wave.unsqueeze(0).unsqueeze(0)  # [1, 1, 24000]

    # -------------------------------------------------------------
    # 2. ENCODEC MODELİ VE RVQ KUANTALAMA
    # -------------------------------------------------------------
    print("\n[2/3] EnCodec (1D Conv + 8 Katmanlı RVQ + 1D Transposed Conv) İle Sıkıştırma...")
    codec = NeuralAudioCodec(in_channels=1, hidden_dim=128, num_quantizers=8, codebook_size=1024)
    with torch.no_grad():
        recon_wave, tokens, commit_loss = codec(audio_wave)

    snr_db = SesMetrikDegerlendirici.snr_hesapla(audio_wave, recon_wave)
    bitrate = SesMetrikDegerlendirici.bitrate_hesapla(kare_orani_hz=75, num_quantizers=8, codebook_bits=10)
    perplexity = SesMetrikDegerlendirici.codebook_perplexity_hesapla(tokens, codebook_size=1024)

    metrikler = {
        "snr_db": snr_db,
        "bitrate_kbps": bitrate,
        "perplexity": perplexity,
    }

    print("\n" + "-" * 80)
    print(f"{'Metrik':<35} | {'Değer'}")
    print("-" * 80)
    print(f"{'Orijinal Ses Boyutu (24 kHz PCM)':<35} | 24,000 Örneklem (1.0 Saniye)")
    print(f"{'RVQ Kuantalama Katmanı (Num_Q)':<35} | 8 Katman")
    print(f"{'Kod Defteri Boyutu (Codebook Size)':<35} | 1,024 Vektör (10 Bit)")
    print(f"{'Ayrık Ses Token Boyutu':<35} | {list(tokens.shape)} [B, Num_Q, T]")
    print(f"{'Hesaplanan Bit Hızı (Bitrate)':<35} | {bitrate} kbps")
    print(f"{'Sinyal-Gürültü Oranı (SNR)':<35} | {snr_db} dB")
    print(f"{'Kod Defteri Çeşitliliği (Perplexity)':<35} | {perplexity}")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Audio Tokenizer Teşhis Panosu Üretiliyor...")
    gorsellestirici = AudioTokenizerGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "audio_tokenizer_encodec_paneli.png")
    gorsellestirici.pano_olustur(audio_wave, recon_wave, metrikler, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 169: NEURAL AUDIO TOKENIZATION BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
