"""
Day 329: Neuromorphic Auditory Cochlea Filters & Event-Based Acoustic Classification
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Gammatone Koklear Filtre Bankasını (ERB Ölçeği), Silikon Koklea Olay Tabanlı İşitsel Dönüştürücüyü
ve Spiking Sinir Ağı (SNN) Akustik Komut Sınıflandırıcısını içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn


class GammatoneFilterBank:
    """
    ERB (Equivalent Rectangular Bandwidth) Logaritmik Frekans Ölçekli Gammatone Filtre Bankası.
    """
    def __init__(self, num_channels: int = 16, f_min: float = 100.0, f_max: float = 6000.0, fs: int = 16000):
        self.num_channels = num_channels
        self.f_min = f_min
        self.f_max = f_max
        self.fs = fs

        # ERB Frekans Aralıklarının Hesaplanması
        erb_min = 21.4 * np.log10(4.37e-3 * f_min + 1.0)
        erb_max = 21.4 * np.log10(4.37e-3 * f_max + 1.0)
        erb_points = np.linspace(erb_min, erb_max, num_channels)
        self.center_freqs = (10.0 ** (erb_points / 21.4) - 1.0) / 4.37e-3

    def filtrele(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Girdi: (Samples,) -> Çıktı: (Num_Channels, Samples) süzülmüş ses sinyali
        """
        num_samples = len(audio_signal)
        t = np.arange(num_samples) / self.fs
        filtered_outputs = np.zeros((self.num_channels, num_samples), dtype=np.float32)

        for c, fc in enumerate(self.center_freqs):
            erb = 24.7 * (4.37e-3 * fc + 1.0)
            b_width = 1.019 * erb
            
            # Gammatone impuls yanıtı: g(t) = t^3 * exp(-2*pi*b*t) * cos(2*pi*fc*t)
            g_t = (t[:128] ** 3) * np.exp(-2 * np.pi * b_width * t[:128]) * np.cos(2 * np.pi * fc * t[:128])
            g_t /= (np.max(np.abs(g_t)) + 1e-9)
            
            # Konvolüsyon ile süzme
            filtered = signal.convolve(audio_signal, g_t, mode="same")
            filtered_outputs[c] = filtered

        return filtered_outputs


class SiliconCochleaEventGenerator:
    """
    Silikon Koklea Olay Tabanlı Sinyal Üreteci (DAS - Dynamic Audio Sensor).
    İç tüy hücreleri yarım dalga doğrultması ve eşik değer uyarlaması ile spike üretir.
    """
    def __init__(self, threshold: float = 0.15, time_bin_size: int = 160):
        self.threshold = threshold
        self.time_bin_size = time_bin_size

    def uret_kokleogram_spikelari(self, filtered_audio: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Girdi: (Channels, Samples) -> Çıktı: (Channels, Time_Bins) Kokleogram Spike Matrisi ve Spike Olay Listesi
        """
        num_channels, total_samples = filtered_audio.shape
        num_bins = total_samples // self.time_bin_size
        
        cochleogram = np.zeros((num_channels, num_bins), dtype=np.float32)
        event_list = []

        # Yarım Dalga Doğrultma (Half-Wave Rectification)
        rectified = np.maximum(0.0, filtered_audio)

        for c in range(num_channels):
            for b in range(num_bins):
                start_idx = b * self.time_bin_size
                end_idx = start_idx + self.time_bin_size
                bin_energy = np.mean(rectified[c, start_idx:end_idx])
                
                if bin_energy > self.threshold:
                    cochleogram[c, b] = 1.0
                    event_list.append((c, b))

        return cochleogram, event_list


class SpikingAudioClassifier(nn.Module):
    """
    Nöromorfik Kokleogram Spike Akışından Akustik Komut Sınıflandırıcısı (SNN).
    Sınıflar: "Evet", "Hayır", "Dur", "Geç" (4 Sınıf)
    """
    def __init__(self, num_channels: int = 16, num_time_bins: int = 100, num_classes: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.num_time_bins = num_time_bins
        
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.lif_mem1 = nn.Parameter(torch.tensor(0.8))
        self.fc = nn.Linear(32 * num_time_bins, num_classes)

    def forward(self, x_cochleogram: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Channels, Time_Bins) -> Çıktı: Logits (Batch, Num_Classes)
        """
        h1 = torch.relu(self.conv1(x_cochleogram))  # (Batch, 32, Time_Bins)
        h1_flat = h1.view(h1.shape[0], -1)
        logits = self.fc(h1_flat)
        return logits
