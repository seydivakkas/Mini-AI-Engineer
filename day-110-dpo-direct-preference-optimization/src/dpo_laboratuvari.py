"""
DPO Hizalama ve Tercih Optimizasyonu Laboratuvarı (Day 110).
Çiftli veri seti üretimi, DPO eğitim döngüsü, örtük ödül takibi ve DPO vs PPO kıyaslaması.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

from .dpo_kaybi import DPOLoss
from .dpo_modeli import DPODilModeli


class DPOLaboratuvari:
    """DPO Eğitim, Örtük Ödül Takip ve Kıyaslama Laboratuvarı."""

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

        # 1. Politika Modeli (Eğitilebilir pi_theta)
        self.policy_model = DPODilModeli(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers
        ).to(cihaz)

        # 2. Referans Model (Dondurulmuş pi_ref)
        self.ref_model = DPODilModeli(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers
        ).to(cihaz)
        self.ref_model.load_state_dict(self.policy_model.state_dict())
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.loss_fn = DPOLoss(beta=0.1, label_smoothing=0.0)

    def sentetik_tercih_verisi_uret(
        self,
        cift_sayisi: int = 300,
        prompt_len: int = 10,
        resp_len: int = 14,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DPO eğitimi için (x, y_w, y_l) ve maske tensörlerini üretir.
        Çıktı: (chosen_ids, rejected_ids, chosen_mask, rejected_mask)
        """
        # Ortak promptlar [B, prompt_len]
        v_min = 1
        v_p_max = max(2, int(self.vocab_size * 0.2))
        v_c_min = v_p_max
        v_c_max = max(v_c_min + 1, int(self.vocab_size * 0.55))
        v_r_min = max(v_c_max, int(self.vocab_size * 0.6))
        v_r_max = self.vocab_size

        prompts = torch.randint(v_min, v_p_max, (cift_sayisi, prompt_len), device=self.cihaz)
        # Tercih edilen kaliteli yanıtlar
        chosen_resp = torch.randint(v_c_min, v_c_max, (cift_sayisi, resp_len), device=self.cihaz)
        # Reddedilen kalitesiz yanıtlar
        rejected_resp = torch.randint(v_r_min, v_r_max, (cift_sayisi, resp_len), device=self.cihaz)

        chosen_ids = torch.cat([prompts, chosen_resp], dim=1)
        rejected_ids = torch.cat([prompts, rejected_resp], dim=1)

        # Yanıt maskesi: Prompt kısmı 0, Yanıt kısmı 1
        chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.float32)
        chosen_mask[:, prompt_len:] = 1.0

        rejected_mask = torch.zeros_like(rejected_ids, dtype=torch.float32)
        rejected_mask[:, prompt_len:] = 1.0

        return chosen_ids, rejected_ids, chosen_mask, rejected_mask

    def dpo_egit(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        epok_sayisi: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        beta: float = 0.1,
    ) -> Dict[str, List[float]]:
        """DPO eğitim döngüsü ve metrik kaydı."""
        self.loss_fn.beta = beta
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)

        N = chosen_ids.shape[0]
        rapor = {
            "kayiplar": [],
            "dogruluklar": [],
            "r_w_ort": [],
            "r_l_ort": [],
            "marjinler": [],
        }

        for ep in range(epok_sayisi):
            self.policy_model.train()
            perm = torch.randperm(N)
            ep_loss, ep_acc, ep_rw, ep_rl, ep_margin = 0.0, 0.0, 0.0, 0.0, 0.0
            adim_sayisi = 0

            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                b_c_ids, b_c_mask = chosen_ids[idx], chosen_mask[idx]
                b_r_ids, b_r_mask = rejected_ids[idx], rejected_mask[idx]

                # Politika modeli log-olasılıkları (Gradyan akışı var)
                pi_logps_chosen = self.policy_model.logprob_hesapla(b_c_ids, b_c_mask)
                pi_logps_rejected = self.policy_model.logprob_hesapla(b_r_ids, b_r_mask)

                # Dondurulmuş referans model log-olasılıkları (Gradyansız)
                with torch.no_grad():
                    ref_logps_chosen = self.ref_model.logprob_hesapla(b_c_ids, b_c_mask)
                    ref_logps_rejected = self.ref_model.logprob_hesapla(b_r_ids, b_r_mask)

                loss, metrikler = self.loss_fn(
                    pi_logps_chosen, pi_logps_rejected, ref_logps_chosen, ref_logps_rejected
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                optimizer.step()

                ep_loss += float(metrikler["kayip"].item())
                ep_acc += float(metrikler["dogruluk"].item())
                ep_rw += float(metrikler["ortuk_odul_chosen"].item())
                ep_rl += float(metrikler["ortuk_odul_rejected"].item())
                ep_margin += float(metrikler["ortuk_marjin"].item())
                adim_sayisi += 1

            rapor["kayiplar"].append(ep_loss / adim_sayisi)
            rapor["dogruluklar"].append((ep_acc / adim_sayisi) * 100.0)
            rapor["r_w_ort"].append(ep_rw / adim_sayisi)
            rapor["r_l_ort"].append(ep_rl / adim_sayisi)
            rapor["marjinler"].append(ep_margin / adim_sayisi)

        return rapor

    def dpo_vs_ppo_kiyasi(self) -> Dict[str, Any]:
        """DPO ile PPO arasındaki VRAM, karmaşıklık ve mimari kıyaslama verileri."""
        return {
            "ppo_model_sayisi": 4,  # Actor, Critic, Ref, RM
            "dpo_model_sayisi": 2,  # Policy, Ref (Critic ve RM YOK!)
            "vram_tasarrufu_yuzde": 50.0,
            "ornekleme_gereksinimi": "DPO: 0 (Off-policy/Kapalı Form), PPO: Yüksek (Rollout Sampling)",
            "egitim_stabilitesi": "DPO: Yüksek (Supervised-like), PPO: Hassas (Hyperparameter sensitive)",
        }
