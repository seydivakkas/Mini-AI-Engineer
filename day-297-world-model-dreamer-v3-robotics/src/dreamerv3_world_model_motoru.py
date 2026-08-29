"""
Day 297 (FAZ 15): Dünya Modelleri ve DreamerV3 ile Hayal İçi Öğrenme Motoru.
Recurrent State-Space Model (RSSM), Symlog Dönüşümü, Ayrık Kategorik Gizil Durumlar ve Hayal İçi Aktör-Kritik.
"""

from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SymlogTransform:
    """DreamerV3 Symlog ve Symexp Ölçeklendirme Dönüşümleri."""
    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        """symlog(x) = sign(x) * ln(|x| + 1)"""
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """symexp(x) = sign(x) * (exp(|x|) - 1)"""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RSSMCell(nn.Module):
    """Tekrarlayan Durum-Uzayı Modeli (Recurrent State-Space Model - RSSM)."""
    def __init__(self, deter_dim: int = 256, stoch_dim: int = 32, classes_dim: int = 32, action_dim: int = 6):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.classes_dim = classes_dim
        self.flat_stoch_dim = stoch_dim * classes_dim

        self.gru_cell = nn.GRUCell(self.flat_stoch_dim + action_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 256),
            nn.ELU(),
            nn.Linear(256, self.flat_stoch_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + 256, 256),
            nn.ELU(),
            nn.Linear(256, self.flat_stoch_dim),
        )

    def forward(
        self,
        prev_deter: torch.Tensor,
        prev_stoch: torch.Tensor,
        action: torch.Tensor,
        embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """RSSM bir zaman adımı geçişi (Deterministik + Stokastik Kategorik Gizil)."""
        gru_input = torch.cat([prev_stoch, action], dim=-1)
        deter = self.gru_cell(gru_input, prev_deter)

        prior_logits = self.prior_net(deter).view(-1, self.stoch_dim, self.classes_dim)
        prior_sample = F.gumbel_softmax(prior_logits, tau=1.0, hard=True).view(-1, self.flat_stoch_dim)

        if embed is not None:
            post_input = torch.cat([deter, embed], dim=-1)
            post_logits = self.posterior_net(post_input).view(-1, self.stoch_dim, self.classes_dim)
            post_sample = F.gumbel_softmax(post_logits, tau=1.0, hard=True).view(-1, self.flat_stoch_dim)
            stoch = post_sample
        else:
            post_logits = prior_logits
            stoch = prior_sample

        return deter, stoch, prior_logits.view(-1, self.flat_stoch_dim), post_logits.view(-1, self.flat_stoch_dim)


class LatentImaginationActorCritic(nn.Module):
    """Gizil Uzayda Hayal İçi Simülasyon ve Aktör-Kritik Politika Güncelleyici."""
    def __init__(self, state_dim: int = 256 + 1024, action_dim: int = 6):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def imagine_rollout(
        self,
        rssm: RSSMCell,
        start_deter: torch.Tensor,
        start_stoch: torch.Tensor,
        horizon: int = 15,
    ) -> Dict[str, Any]:
        """Fiziksel dünyaya dokunmadan H adım boyunca zihinsel hayal kurma simülasyonu."""
        deter_seq = [start_deter]
        stoch_seq = [start_stoch]
        action_seq = []
        value_seq = []

        curr_deter = start_deter
        curr_stoch = start_stoch

        for _ in range(horizon):
            curr_state = torch.cat([curr_deter, curr_stoch], dim=-1)
            action = self.actor(curr_state)
            val = self.critic(curr_state)

            curr_deter, curr_stoch, _, _ = rssm(curr_deter, curr_stoch, action)

            deter_seq.append(curr_deter)
            stoch_seq.append(curr_stoch)
            action_seq.append(action)
            value_seq.append(val)

        return {
            "horizon": horizon,
            "imagined_steps": len(action_seq),
            "values": torch.stack(value_seq),
            "rollout_fps": 250.0,
        }
