"""
Ses Kalitesi ve Sıkıştırma Metrikleri Değerlendirici (Day 169 - FAZ 9).
SNR (Signal-to-Noise Ratio), SI-SDR, Kod Defteri Yoğunluğu (Perplexity) ve Bitrate (kbps) hesaplar.
"""

from typing import Dict, Any
import torch
import numpy as np


class SesMetrikDegerlendirici:
    """Ses Sıkıştırma Başarısını Ölçen Metrik Motoru."""

    @classmethod
    def snr_hesapla(cls, referans: torch.Tensor, tahmin: torch.Tensor) -> float:
        """
        Sinyal-Gürültü Oranı (Signal-to-Noise Ratio - SNR in dB):
        SNR = 10 * log10( sum(ref^2) / sum((ref - est)^2) )
        """
        sinyal_gucu = torch.sum(referans ** 2)
        gurultu_gucu = torch.sum((referans - tahmin) ** 2) + 1e-8
        snr_db = 10.0 * torch.log10(sinyal_gucu / gurultu_gucu).item()
        return round(float(snr_db), 2)

    @classmethod
    def bitrate_hesapla(
        cls,
        ornekleme_frekansi_hz: int = 24000,
        kare_orani_hz: int = 75,
        num_quantizers: int = 8,
        codebook_bits: int = 10,  # 1024 = 2^10
    ) -> float:
        """
        Bit Hızı (Bitrate in kbps):
        Bitrate = (kare_orani * num_quantizers * codebook_bits) / 1000
        Örn: 75 fps * 8 Q * 10 bit = 6.0 kbps
        """
        toplam_bps = kare_orani_hz * num_quantizers * codebook_bits
        return round(toplam_bps / 1000.0, 2)

    @classmethod
    def codebook_perplexity_hesapla(cls, tokens: torch.Tensor, codebook_size: int = 1024) -> float:
        """
        Kod Defteri Kullanım Çeşitliliği (Perplexity):
        Tüm indekslerin dağılım entropisi ve perplexity değeri.
        """
        flat_tokens = tokens.flatten().cpu().numpy()
        counts = np.bincount(flat_tokens, minlength=codebook_size)
        probs = counts / (np.sum(counts) + 1e-8)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        perplexity = 2 ** entropy
        return round(float(perplexity), 2)
