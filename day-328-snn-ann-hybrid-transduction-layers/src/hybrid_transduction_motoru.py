"""
Day 328: SNN-ANN Hybrid Transduction Layers
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; ANN sürekli aktivasyonlarını SNN zamansal spike akışlarına çeviren ANN-to-SNN Transducer,
Spike akışlarını pürüzsüz ANN vektörlerine çeviren SNN-to-ANN Transducer ve Hibrit Derin Ağ Mimarisini içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Spike eşik fonksiyonu için türevlenebilir surrogate gradient.
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return (input_tensor >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output / (1.0 + alpha * torch.abs(input_tensor)) ** 2
        return grad_input, None


class ANNToSNNTransducer(nn.Module):
    """
    ANN-to-SNN Dönüştürücü Katman (Transduction Layer).
    Sürekli ANN aktivasyonlarını (B, N_ann) -> Zamansal Spike Akışına (B, Time, N_snn) çevirir.
    """
    def __init__(self, in_features: int, out_features: int, time_steps: int = 10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_steps = time_steps
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x_ann: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, N_ann) -> Çıktı: (Batch, Time, N_snn)
        """
        b_size = x_ann.shape[0]
        prob = torch.sigmoid(self.proj(x_ann))  # (B, N_snn) in [0, 1]

        # Time adımları boyunca Poisson Spike Üretimi
        prob_expanded = prob.unsqueeze(1).repeat(1, self.time_steps, 1)  # (B, Time, N_snn)
        random_noise = torch.rand_like(prob_expanded)
        spike_stream = (random_noise < prob_expanded).float()
        return spike_stream


class SNNToANNTransducer(nn.Module):
    """
    SNN-to-ANN Dönüştürücü Katman (Transduction Layer).
    Zamansal 1-bit Spike Akışını (B, Time, N_snn) -> Pürüzsüz Sürekli ANN Vektörüne (B, N_ann) çevirir.
    Düşük Geçiren Filtre (Low-Pass Filter Decay) ile hesaplanır.
    """
    def __init__(self, in_features: int, out_features: int, tau_decay: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_decay = tau_decay
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, spike_stream: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Time, N_snn) -> Çıktı: (Batch, N_ann)
        """
        b_size, t_steps, n_snn = spike_stream.shape
        t_indices = torch.arange(t_steps, device=spike_stream.device, dtype=torch.float32)
        weights = torch.exp(-(t_steps - 1 - t_indices) / self.tau_decay)  # (Time,)
        weights = (weights / weights.sum()).view(1, t_steps, 1)

        # Üstel düşük geçiren süzgeç ile entegrasyon
        filtered_spikes = torch.sum(spike_stream * weights, dim=1)  # (B, N_snn)
        x_ann_out = F.relu(self.proj(filtered_spikes))               # (B, N_ann)
        return x_ann_out


class SNNLIFLayer(nn.Module):
    """
    Spiking Neural Network Leaky Integrate-and-Fire (LIF) Katmanı.
    """
    def __init__(self, num_neurons: int, beta: float = 0.8, v_th: float = 1.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.beta = beta
        self.v_th = v_th

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Girdi: (Batch, Time, Neurons) -> Çıktı: (out_spikes, v_mem_history)
        """
        b_size, t_steps, _ = input_spikes.shape
        v_mem = torch.zeros(b_size, self.num_neurons, device=input_spikes.device)
        
        out_spikes_list = []
        v_mem_list = []

        for t in range(t_steps):
            inp = input_spikes[:, t, :]
            v_mem = self.beta * v_mem + inp
            
            # Spike Tetikleme
            spike = FastSigmoidSurrogate.apply(v_mem - self.v_th)
            v_mem = v_mem * (1.0 - spike)  # Soft Reset
            
            out_spikes_list.append(spike)
            v_mem_list.append(v_mem)

        out_spikes = torch.stack(out_spikes_list, dim=1)
        v_mem_history = torch.stack(v_mem_list, dim=1)
        return out_spikes, v_mem_history


class HybridSNNANNNetwork(nn.Module):
    """
    ANN-SNN-ANN Hibrit Derin Ağ Mimarisi.
    
    Akış:
        1. ANN Giriş Katmanı (Linear/ReLU)
        2. ANN-to-SNN Transducer (Spike Stream)
        3. SNN LIF Spiking Layer (Ultra-Low Power SOP)
        4. SNN-to-ANN Transducer (Low-Pass Filter)
        5. ANN Çıkış Katmanı (Linear Classifier)
    """
    def __init__(
        self,
        in_features: int = 64,
        ann_hidden: int = 32,
        snn_neurons: int = 32,
        num_classes: int = 4,
        time_steps: int = 10
    ):
        super().__init__()
        self.ann_input = nn.Sequential(
            nn.Linear(in_features, ann_hidden),
            nn.ReLU()
        )
        self.ann_to_snn = ANNToSNNTransducer(in_features=ann_hidden, out_features=snn_neurons, time_steps=time_steps)
        self.snn_layer = SNNLIFLayer(num_neurons=snn_neurons, beta=0.8, v_th=1.0)
        self.snn_to_ann = SNNToANNTransducer(in_features=snn_neurons, out_features=ann_hidden, tau_decay=2.0)
        self.classifier = nn.Linear(ann_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Çıktı: (logits, spike_stream, v_mem_history)
        """
        h_ann1 = self.ann_input(x)
        spike_stream_in = self.ann_to_snn(h_ann1)
        out_spikes, v_mem_history = self.snn_layer(spike_stream_in)
        h_ann2 = self.snn_to_ann(out_spikes)
        logits = self.classifier(h_ann2)
        return logits, out_spikes, v_mem_history
