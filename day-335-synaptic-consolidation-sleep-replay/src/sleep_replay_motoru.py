"""
Day 335: Synaptic Consolidation & Sleep Replay (Zero Catastrophic Forgetting)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Sinaptik Etiketleme ve Konsolidasyonu (STC), Hipokampal Uyku Fazı Tekrarını (Sleep Replay)
ve Yıkıcı Unutmayı Engelleyen Sürekli Öğrenme (Continual Learning) Ağını içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SynapticTaggingConsolidator:
    """
    Sinaptik Etiketleme ve Konsolidasyon (Synaptic Tagging & Capture - STC) Motoru.
    Fisher Bilgi Matrisi (Fisher Information Matrix) ile önemli ağırlıkları korur.
    """
    def __init__(self, model: nn.Module, lambda_cons: float = 500.0):
        self.model = model
        self.lambda_cons = lambda_cons
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_weights: Dict[str, torch.Tensor] = {}

    def compute_fisher_information(self, data_loader: torch.utils.data.DataLoader, criterion: nn.Module):
        """
        Task 1 verisi üzerinden her bir ağırlığın önem derecesini (Fisher Information) hesaplar.
        """
        self.model.eval()
        self.fisher_dict = {}
        self.optimal_weights = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param)
                self.optimal_weights[name] = param.data.clone()

        for inputs, targets in data_loader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_dict[name] += (param.grad.data ** 2) / len(data_loader)

    def consolidation_loss(self) -> torch.Tensor:
        """
        Konsolide edilmiş eski ağırlıklardan sapmayı cezalandıran EWC kaybı.
        """
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                opt_w = self.optimal_weights[name]
                loss += torch.sum(fisher * ((param - opt_w) ** 2))
        return (self.lambda_cons / 2.0) * loss


class HippocampalSleepReplayer:
    """
    Hipokampal Uyku Fazı Bellek Tekrarı (Slow-Wave Sleep Replay) Tamponu.
    Gündüz öğrenilen spike izlerini gece hızlandırılmış tekrarlarla beyin kabuğuna aktarır.
    """
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.replay_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def store_wake_memory(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Gündüz waking verisini tekil örnekler halinde belleğe kaydeder."""
        if inputs.ndim == 1:
            if len(self.replay_buffer) < self.capacity:
                self.replay_buffer.append((inputs.detach().clone(), targets.detach().clone()))
        else:
            for i in range(inputs.shape[0]):
                if len(self.replay_buffer) < self.capacity:
                    self.replay_buffer.append((inputs[i].detach().clone(), targets[i].detach().clone()))

    def sample_sleep_replay(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uyku fazında rastgele bellek tekrarı verisi örnekler."""
        num_samples = min(batch_size, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), size=num_samples, replace=False)
        batch_x = torch.stack([self.replay_buffer[i][0] for i in indices])
        batch_y = torch.stack([self.replay_buffer[i][1] for i in indices])
        return batch_x, batch_y


class ContinualSpikingNetwork(nn.Module):
    """
    Sıfır Yıkıcı Unutma (Zero Catastrophic Forgetting) Hedefli Sürekli Öğrenme Ağı.
    """
    def __init__(self, input_dim: int = 20, hidden_dim: int = 32, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        out = self.fc2(h)
        return out
