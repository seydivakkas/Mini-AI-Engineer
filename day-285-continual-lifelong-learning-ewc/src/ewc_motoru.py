"""
Day 285 (FAZ 15): Sürekli ve Yaşam Boyu Öğrenme (Continual Learning) Motoru.
Elastic Weight Consolidation (EWC), Fisher Bilgi Matrisi ve Synaptic Intelligence.
"""

from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleClassifier(nn.Module):
    """Sürekli Öğrenme Çok Görevli Yapay Sinir Ağı."""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ContinualLifelongLearningEngine:
    """
    FAZ 15 Sürekli ve Yaşam Boyu Öğrenme Motoru.
    
    Özellikler:
    - Fisher Bilgi Matrisi (Diagonal Fisher Information Matrix: F_i)
    - Elastic Weight Consolidation (EWC) Karesel Ceza Kaybı
    - Yıkıcı Unutmayı Önleme (Catastrophic Forgetting Mitigation)
    - Synaptic Intelligence (SI) Ağırlık Yörüngesi Önem Puanlaması
    """

    @classmethod
    def compute_fisher_information(
        cls,
        model: nn.Module,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Görev A Üzerinde Çapraz Fisher Bilgi Matrisinin (F_i) Hesaplanması:
        F_i = (1/N) * sum_k [ (grad_theta_i log p(y_k | x_k, theta))^2 ]
        """
        model.eval()
        fisher_dict = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}
        criterion = nn.CrossEntropyLoss()

        total_samples = len(dataset)
        for x, y in dataset:
            model.zero_grad()
            output = model(x.unsqueeze(0))
            loss = criterion(output, y.unsqueeze(0))
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += (param.grad.data ** 2) / total_samples

        return fisher_dict

    @classmethod
    def compute_ewc_loss(
        cls,
        model: nn.Module,
        task_loss: torch.Tensor,
        optimal_params: Dict[str, torch.Tensor],
        fisher_dict: Dict[str, torch.Tensor],
        lambda_ewc: float = 5000.0,
    ) -> torch.Tensor:
        """
        EWC Kayıp Fonksiyonu:
        L(θ) = L_B(θ) + sum_i (λ / 2) * F_i * (θ_i - θ_A,i*)^2
        """
        ewc_penalty = torch.tensor(0.0, device=task_loss.device)
        for name, param in model.named_parameters():
            if name in optimal_params and name in fisher_dict:
                f_i = fisher_dict[name]
                theta_star = optimal_params[name]
                ewc_penalty += torch.sum(f_i * ((param - theta_star) ** 2))

        total_loss = task_loss + (lambda_ewc / 2.0) * ewc_penalty
        return total_loss

    @classmethod
    def evaluate_accuracy(
        cls,
        model: nn.Module,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        """Doğruluk yüzdesini (%) ölçer."""
        model.eval()
        correct = 0
        total = len(dataset)
        with torch.no_grad():
            for x, y in dataset:
                out = model(x.unsqueeze(0))
                pred = torch.argmax(out, dim=1).item()
                if pred == y.item():
                    correct += 1
        return (correct / total) * 100.0
