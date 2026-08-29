"""
Teleoperasyon ve Taklit Öğrenmesi (ACT - Action Chunking with Transformers) Motoru (Day 255).
CVAE Latent Niyet Kodlama, K-Adımlı Eylem Yığını Tahmini ve Zamansal Topluluk (Temporal Ensembling).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeleoperationDataBuffer:
    """Teleoperasyon İnsan Demonstrasyon Kayıt ve Eylem Yığını (Chunking) Havuzu."""

    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.samples = []

    def add_demonstration(self, states: np.ndarray, actions: np.ndarray):
        """Birleşik durum ve eylem yörüngesini K uzunluklu kayan pencerelere böler."""
        t_len = len(actions)
        for t in range(t_len):
            s_t = states[t]
            # K boyutlu eylem yığını (Chunk)
            if t + self.chunk_size <= t_len:
                chunk = actions[t : t + self.chunk_size]
            else:
                # Son adımlarda son eylemi tekrarla (padding)
                pad_len = (t + self.chunk_size) - t_len
                chunk = np.vstack([actions[t:], np.tile(actions[-1], (pad_len, 1))])

            self.samples.append({
                "state": s_t,
                "action_chunk": chunk,
            })

    def get_batch(self, batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eğitim için rastgele mini-batch çeker."""
        indices = np.random.choice(len(self.samples), size=min(batch_size, len(self.samples)), replace=False)
        states = [self.samples[i]["state"] for i in indices]
        chunks = [self.samples[i]["action_chunk"] for i in indices]

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(chunks), dtype=torch.float32),
        )


class ACTCVAEModel(nn.Module):
    """Action Chunking with Transformers (ACT) ve CVAE Eylem Yığını Tahmincisi."""

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 7,
        chunk_size: int = 10,
        latent_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # 1. CVAE Kodlayıcı: (s_t, A_t) -> (mu, log_std)
        flat_action_dim = chunk_size * action_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + flat_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_std = nn.Linear(hidden_dim, latent_dim)

        # 2. ACT Dekoder: (s_t, z) -> K x action_dim
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Reparameterization Trick: z = mu + std * eps."""
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        states: torch.Tensor,
        action_chunks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """İleri yayılım: Eğitimde CVAE z örnekler, çıkarımda z=0 ortalamasını kullanır."""
        batch_size = states.shape[0]

        if action_chunks is not None:
            # Eğitim Modu (CVAE Encoder aktif)
            flat_act = action_chunks.view(batch_size, -1)
            enc_input = torch.cat([states, flat_act], dim=-1)
            h = self.encoder(enc_input)
            mu = self.fc_mu(h)
            log_std = self.fc_log_std(h)
            z = self.reparameterize(mu, log_std)
        else:
            # Çıkarım / Test Modu (Latent prior z = 0)
            mu = torch.zeros(batch_size, self.latent_dim, device=states.device)
            log_std = torch.zeros(batch_size, self.latent_dim, device=states.device)
            z = mu

        # Dekoder
        dec_input = torch.cat([states, z], dim=-1)
        pred_flat = self.decoder(dec_input)
        pred_chunks = pred_flat.view(batch_size, self.chunk_size, self.action_dim)

        return pred_chunks, mu, log_std

    def compute_loss(
        self,
        pred_chunks: torch.Tensor,
        target_chunks: torch.Tensor,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        kl_weight: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """L1 Yeniden Yapılandırma Kaybı + KL Iraksama Kaybı."""
        l1_loss = F.l1_loss(pred_chunks, target_chunks)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_std - mu.pow(2) - log_std.exp(), dim=-1))
        total_loss = l1_loss + kl_weight * kl_loss

        return {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "kl_loss": kl_loss,
        }


class TemporalEnsembler:
    """K-Adımlı Çakışan Eylem Tahminlerini Üstel Ağırlıkla Yumuşatıcı (Temporal Ensemble)."""

    def __init__(self, chunk_size: int = 10, m_decay: float = 0.05):
        self.chunk_size = chunk_size
        self.m = m_decay
        self.weights = np.exp(-self.m * np.arange(self.chunk_size))
        self.history = []  # Geçmiş eylem yığınları

    def add_prediction(self, pred_chunk_np: np.ndarray):
        """Yeni üretilen K boyutlu eylem tahminini kaydeder."""
        self.history.append(pred_chunk_np.copy())
        if len(self.history) > self.chunk_size:
            self.history.pop(0)

    def get_ensembled_action(self) -> np.ndarray:
        """Şimdiki an için çakışan tahminlerin ağırlıklı ortalamasını döner."""
        num_chunks = len(self.history)
        action_dim = self.history[0].shape[1]

        total_act = np.zeros(action_dim, dtype=np.float64)
        total_weight = 0.0

        for i in range(num_chunks):
            # i. geçmiş yığının (num_chunks - 1 - i). adımı şimdiki zamana denk gelir
            chunk_idx = num_chunks - 1 - i
            act_i = self.history[i][chunk_idx]
            w_i = self.weights[chunk_idx]

            total_act += w_i * act_i
            total_weight += w_i

        ensembled_action = total_act / total_weight
        return ensembled_action.round(4)
