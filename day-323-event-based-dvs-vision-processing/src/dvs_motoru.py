"""
Day 323: Dynamic Vision Sensors (DVS) & Event-Based Processing
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; DVS (Dynamic Vision Sensors) nöromorfik kamera olay akışı simülasyonunu,
Surface of Active Events (SAE) zamansal sönümlenme yüzeyini, 3D Voxel Grid kodlayıcısını
ve olay tabanlı Spiking ConvNet modelini barındırır.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Spiking ConvNet için türevlenebilir Fast Sigmoid Surrogate Gradient.
    """
    slope: float = 25.0

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, v_th: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.v_th = v_th
        return (input_tensor >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        v_th = ctx.v_th
        slope = FastSigmoidSurrogate.slope
        grad_input = grad_output.clone()
        surrogate_grad = slope / ((1.0 + slope * torch.abs(input_tensor - v_th)) ** 2)
        return grad_input * surrogate_grad, None


class DVSEventStreamGenerator:
    """
    Nöromorfik DVS Kamera Olay Akışı Simülatörü.
    Mikrosaniye (us) zaman damgalı olay kümesi e_k = (x, y, t_us, p) üretir.
    """
    def __init__(self, height: int = 32, width: int = 32):
        self.height = height
        self.width = width

    def uret_hareketli_cizgi(
        self,
        yon: str = "sag",
        duration_us: int = 50000,
        num_events: int = 1200,
        seed: int = 42
    ) -> np.ndarray:
        """
        'sag', 'sol', 'yukari', 'asagi' yönünde hareket eden çizgi nesnesinden
        (N, 4) boyutlu DVS olay akışı üretir: [x, y, t_us, polarity]
        """
        np.random.seed(seed)
        t_stamps = np.sort(np.random.uniform(0, duration_us, num_events))
        events = np.zeros((num_events, 4), dtype=np.float32)

        for i, t in enumerate(t_stamps):
            progress = t / duration_us  # 0.0 -> 1.0
            
            if yon == "sag":
                x = int(progress * (self.width - 5)) + 2
                y = int(np.random.uniform(2, self.height - 3))
            elif yon == "sol":
                x = int((1.0 - progress) * (self.width - 5)) + 2
                y = int(np.random.uniform(2, self.height - 3))
            elif yon == "yukari":
                x = int(np.random.uniform(2, self.width - 3))
                y = int((1.0 - progress) * (self.height - 5)) + 2
            else:  # asagi
                x = int(np.random.uniform(2, self.width - 3))
                y = int(progress * (self.height - 5)) + 2

            # Polarite +1 (Parlaklık artışı) veya -1 (Parlaklık azalışı)
            pol = 1.0 if np.random.rand() > 0.3 else -1.0
            
            # Gürültü ekle
            x = np.clip(x + np.random.randint(-1, 2), 0, self.width - 1)
            y = np.clip(y + np.random.randint(-1, 2), 0, self.height - 1)

            events[i] = [x, y, t, pol]

        return events


class SurfaceOfActiveEvents:
    """
    Surface of Active Events (SAE) / Zamansal Sönümlenme Yüzeyi.

    Matematiksel Formül:
        S(x, y, p) = exp( -(t_current - T_last(x, y, p)) / tau )
    """
    def __init__(self, height: int = 32, width: int = 32, tau_us: float = 10000.0):
        self.height = height
        self.width = width
        self.tau_us = tau_us
        # T_last(x, y, polarity_idx) -> polarity: 0 (-1), 1 (+1)
        self.t_last = np.zeros((2, height, width), dtype=np.float32)

    def guncelle_ve_hesapla(self, events: np.ndarray, t_current: float) -> np.ndarray:
        """
        Olaylar ile T_last matrisini günceller ve üstel sönümlenmiş SAE yüzeyini döner.
        Girdi: events (N, 4) -> [x, y, t_us, polarity]
        Çıktı: sae_surface (2, Height, Width)
        """
        for x, y, t, p in events:
            ix, iy = int(x), int(y)
            p_idx = 1 if p > 0 else 0
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.t_last[p_idx, iy, ix] = t

        # Üstel Sönümlenme Hesabı
        delta_t = t_current - self.t_last
        delta_t = np.maximum(delta_t, 0.0)
        
        # Eğer hiç olay olmamışsa (t_last == 0), SAE = 0 olsun
        sae = np.where(self.t_last > 0, np.exp(-delta_t / self.tau_us), 0.0)
        return sae.astype(np.float32)


class VoxelGridEncoder:
    """
    DVS Olay Akışını 3D Voxel Grid Tensörüne Dönüştürücü.
    Girdi olaylarını zamansal B kutusuna (bins) böler.
    Çıktı Tensörü Boyutu: (2 * num_bins, Height, Width)
    """
    def __init__(self, height: int = 32, width: int = 32, num_bins: int = 5):
        self.height = height
        self.width = width
        self.num_bins = num_bins

    def kodla(self, events: np.ndarray, duration_us: float = 50000.0) -> torch.Tensor:
        """
        Olay akışından PyTorch Voxel Grid Tensörü (C=2*num_bins, H, W) üretir.
        """
        voxel_grid = torch.zeros((2 * self.num_bins, self.height, self.width), dtype=torch.float32)
        if len(events) == 0:
            return voxel_grid

        t_min = 0.0
        t_max = duration_us

        for x, y, t, p in events:
            ix, iy = int(x), int(y)
            if not (0 <= ix < self.width and 0 <= iy < self.height):
                continue
                
            # Normalize zaman t_norm in [0, num_bins - 1]
            t_norm = (t - t_min) / (t_max - t_min + 1e-9) * (self.num_bins - 1)
            t_norm = max(0.0, min(float(self.num_bins - 1), float(t_norm)))
            
            bin_idx = int(t_norm)
            pol_offset = 0 if p > 0 else self.num_bins
            channel = pol_offset + bin_idx

            voxel_grid[channel, iy, ix] += 1.0

        return voxel_grid


class SpikingEventConvNet(nn.Module):
    """
    DVS Voxel Grid Girdileri İçin Olay Tabanlı Spiking Evrişimsel Ağ (Spiking ConvNet).
    
    Mimari:
        Conv2D -> LIF Activation -> Conv2D -> LIF Activation -> AdaptiveAvgPool -> Linear Readout
    """
    def __init__(self, in_channels: int = 10, num_classes: int = 4, beta: float = 0.85, v_th: float = 1.0):
        super().__init__()
        self.beta = beta
        self.v_th = v_th

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Girdi: Voxel Grid Tensörü (Batch, In_Channels, H, W)
        """
        batch_size = x.shape[0]

        # Katman 1
        h1 = self.conv1(x)
        s1 = FastSigmoidSurrogate.apply(h1, self.v_th)

        # Katman 2
        h2 = self.conv2(s1)
        s2 = FastSigmoidSurrogate.apply(h2, self.v_th)

        # Havuzlama & Sınıflandırma
        pooled = self.pool(s2)
        flat = pooled.view(batch_size, -1)
        logits = self.fc(flat)

        info_dict = {
            "spikes_l1": s1,
            "spikes_l2": s2,
            "sparsity_l1": torch.mean(s1).item(),
            "sparsity_l2": torch.mean(s2).item(),
        }
        return logits, info_dict
