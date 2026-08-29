"""
Sinirsel Ses Sıkıştırma (Neural Audio Codec / EnCodec) Modülü (Day 169 - FAZ 9).
1D Conv Encoder + RVQ + 1D Transposed Conv Decoder mimarisi.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from .residual_vector_quantizer import ResidualVectorQuantizer


class NeuralAudioCodec(nn.Module):
    """EnCodec ve SoundStream Tarzı Sinirsel Ses Kodlayıcı / Kod Çözücü."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 128,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 1. 1D Konvülsiyonel Kodlayıcı (Encoder)
        # Örnekleme oranını 4 kat azaltır (Strided 1D Conv)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ELU(),
            nn.Conv1d(64, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.ELU(),
        )

        # 2. RVQ (Residual Vector Quantizer)
        self.rvq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            dim=hidden_dim,
        )

        # 3. 1D Transposed Conv Kod Çözücü (Decoder)
        # Sinyali tekrar orijinal uzunluğa genişletir
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(64, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh(),  # Ses dalgası [-1.0, 1.0] aralığında
        )

    def forward(self, audio_wave: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        audio_wave: [B, 1, Ses_Uzunlugu]
        Döner: (reconstructed_wave, discrete_tokens, rvq_loss)
        """
        # Encoder: [B, C, L] -> [B, D, T]
        z = self.encoder(audio_wave)
        z = z.permute(0, 2, 1)  # [B, T, D]

        # RVQ Kuantalama: [B, T, D] -> [B, T, D], [B, Num_Q, T]
        z_q, tokens, commit_loss = self.rvq(z)

        # Decoder: [B, T, D] -> [B, D, T] -> [B, 1, L]
        z_q = z_q.permute(0, 2, 1)
        recon_wave = self.decoder(z_q)

        # Uzunluk eşleme (padding düzeltmesi)
        if recon_wave.shape[-1] != audio_wave.shape[-1]:
            recon_wave = recon_wave[..., :audio_wave.shape[-1]]

        return recon_wave, tokens, commit_loss
