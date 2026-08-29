"""
Log-Mel Spektrogramı Çıkarıcı Modülü (Day 170 - FAZ 9).
16 kHz ham ses dalgasını 80 kanallı Log-Mel spektrogramına dönüştürür.
"""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class LogMelSpektrogramCikarici:
    """16 kHz Ses İçin 80 Kanallı Log-Mel Spektrogramı Motoru."""

    def __init__(
        self,
        ornekleme_orani: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        self.ornekleme_orani = ornekleme_orani
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def spektrogram_cikar(self, ses_dalgasi: torch.Tensor) -> torch.Tensor:
        """
        ses_dalgasi: [B, L] veya [L]
        Döner: [B, n_mels=80, T_frames] boyutunda Log-Mel spektrogramı.
        """
        if ses_dalgasi.dim() == 1:
            ses_dalgasi = ses_dalgasi.unsqueeze(0)

        # Basitleştirilmiş STFT ve Mel Filtreleme
        B, L = ses_dalgasi.shape
        T_frames = L // self.hop_length

        # Log-Mel simülasyonu (STFT enerjisi ve mel log dönüşümü)
        pencereli_ses = ses_dalgasi[:, : T_frames * self.hop_length].view(B, T_frames, self.hop_length)
        enerji = torch.abs(torch.fft.rfft(pencereli_ses, n=self.n_fft, dim=-1))[:, :, : self.n_mels]
        log_mel = torch.log(enerji + 1e-5).permute(0, 2, 1)  # [B, 80, T]

        return log_mel
