"""
ORPO Hizalama ve Monolitik SFT Laboratuvarı (Day 112).
Tek aşamalı SFT + Alignment eğitimi, Log-Odds dinamiği ve 1-Aşama vs 2-Aşama kıyası.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

from .orpo_kaybi import ORPOLoss
from .orpo_modeli import ORPODilModeli


class ORPOLaboratuvari:
    """ORPO Tek Aşamalı Monolitik Hizalama Laboratuvarı."""

    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        cihaz: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.cihaz = cihaz

        # Yalnızca TEK BİR model! (Referans model veya Critic YOK!)
        self.model = ORPODilModeli(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers
        ).to(cihaz)

        self.loss_fn = ORPOLoss(lambda_or=0.5)

    def sentetik_tercih_verisi_uret(
        self,
        cift_sayisi: int = 350,
        prompt_len: int = 10,
        resp_len: int = 14,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ORPO eğitimi için (x, y_w, y_l) ve maskeleri üretir.
        Çıktı: (chosen_ids, rejected_ids, chosen_mask, rejected_mask)
        """
        v_p_max = max(2, int(self.vocab_size * 0.2))
        v_c_min = v_p_max
        v_c_max = max(v_c_min + 1, int(self.vocab_size * 0.55))
        v_r_min = max(v_c_max, int(self.vocab_size * 0.6))
        v_r_max = self.vocab_size

        prompts = torch.randint(1, v_p_max, (cift_sayisi, prompt_len), device=self.cihaz)
        chosen_resp = torch.randint(v_c_min, v_c_max, (cift_sayisi, resp_len), device=self.cihaz)
        rejected_resp = torch.randint(v_r_min, v_r_max, (cift_sayisi, resp_len), device=self.cihaz)

        chosen_ids = torch.cat([prompts, chosen_resp], dim=1)
        rejected_ids = torch.cat([prompts, rejected_resp], dim=1)

        chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.float32)
        chosen_mask[:, prompt_len:] = 1.0

        rejected_mask = torch.zeros_like(rejected_ids, dtype=torch.float32)
        rejected_mask[:, prompt_len:] = 1.0

        return chosen_ids, rejected_ids, chosen_mask, rejected_mask

    def orpo_egit(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        epok_sayisi: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        lambda_or: float = 0.5,
    ) -> Dict[str, List[float]]:
        """ORPO monolitik eğitim döngüsü ve metrik kaydı."""
        self.loss_fn.lambda_or = lambda_or
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        N = chosen_ids.shape[0]
        rapor = {
            "toplam_kayiplar": [],
            "kayiplar_sft": [],
            "kayiplar_or": [],
            "log_odds_oranlari": [],
            "dogruluklar": [],
        }

        for ep in range(epok_sayisi):
            self.model.train()
            perm = torch.randperm(N)
            ep_loss, ep_sft, ep_or, ep_odds, ep_acc = 0.0, 0.0, 0.0, 0.0, 0.0
            adim_sayisi = 0

            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                b_c_ids, b_c_mask = chosen_ids[idx], chosen_mask[idx]
                b_r_ids, b_r_mask = rejected_ids[idx], rejected_mask[idx]

                # Tekil model üzerinden token log-olasılıkları
                chosen_logps = self.model.token_logprob_hesapla(b_c_ids)
                rejected_logps = self.model.token_logprob_hesapla(b_r_ids)

                loss, metrikler = self.loss_fn(
                    chosen_logps, rejected_logps, b_c_mask, b_r_mask
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                ep_loss += float(metrikler["toplam_kayip"].item())
                ep_sft += float(metrikler["kayip_sft"].item())
                ep_or += float(metrikler["kayip_or"].item())
                ep_odds += float(metrikler["log_odds_ratio"].item())
                ep_acc += float(metrikler["dogruluk"].item())
                adim_sayisi += 1

            rapor["toplam_kayiplar"].append(ep_loss / adim_sayisi)
            rapor["kayiplar_sft"].append(ep_sft / adim_sayisi)
            rapor["kayiplar_or"].append(ep_or / adim_sayisi)
            rapor["log_odds_oranlari"].append(ep_odds / adim_sayisi)
            rapor["dogruluklar"].append((ep_acc / adim_sayisi) * 100.0)

        return rapor

    def mimari_kiyas_raporu(self) -> Dict[str, Any]:
        """PPO, DPO ve ORPO arasındaki mimari ve boru hattı kıyaslaması."""
        return {
            "ppo_asamalar": "3 Aşama (SFT -> RM -> PPO RL)",
            "dpo_asamalar": "2 Aşama (SFT -> DPO Alignment)",
            "orpo_asamalar": "1 Aşama (Monolitik SFT + Alignment)",
            "ppo_gpu_model": 4,  # Actor, Critic, Ref, RM
            "dpo_gpu_model": 2,  # Policy, Ref
            "orpo_gpu_model": 1,  # Yalnızca 1 Model!
            "orpo_vram_tasarrufu_dpo_gore": "%50 Daha Az VRAM",
            "orpo_vram_tasarrufu_ppo_gore": "%75 Daha Az VRAM",
        }
