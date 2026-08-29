"""
SimPO Tercih Laboratuvarı ve 4'lü Hizalama Kıyaslama Motoru (Day 113).
Referans modelsiz SimPO eğitimi, marjin dinamiği ve PPO vs DPO vs ORPO vs SimPO analizi.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

from .simpo_kaybi import SimPOLoss
from .simpo_modeli import SimPODilModeli


class SimPOLaboratuvari:
    """SimPO Tercih Optimizasyonu ve Marjin Analiz Laboratuvarı."""

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

        # Referans Modelsiz TEKİL Politika Modeli
        self.model = SimPODilModeli(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers
        ).to(cihaz)

        self.loss_fn = SimPOLoss(beta=2.0, gamma=0.5)

    def sentetik_tercih_verisi_uret(
        self,
        cift_sayisi: int = 350,
        prompt_len: int = 10,
        resp_len: int = 14,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SimPO eğitimi için (x, y_w, y_l) çiftlerini ve yanıt maskelerini üretir."""
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

    def simpo_egit(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        epok_sayisi: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        beta: float = 2.0,
        gamma: float = 0.5,
    ) -> Dict[str, List[float]]:
        """SimPO eğitim döngüsü ve hedef marjin metrik kaydı."""
        self.loss_fn.beta = beta
        self.loss_fn.gamma = gamma
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        N = chosen_ids.shape[0]
        rapor = {
            "kayiplar": [],
            "chosen_odulleri": [],
            "rejected_odulleri": [],
            "odul_farklari": [],
            "marjin_ihlalleri": [],
            "dogruluklar": [],
        }

        for ep in range(epok_sayisi):
            self.model.train()
            perm = torch.randperm(N)
            ep_loss, ep_cw, ep_cr, ep_df, ep_mv, ep_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            adim_sayisi = 0

            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                b_c_ids, b_c_mask = chosen_ids[idx], chosen_mask[idx]
                b_r_ids, b_r_mask = rejected_ids[idx], rejected_mask[idx]

                chosen_logps = self.model.token_logprob_hesapla(b_c_ids)
                rejected_logps = self.model.token_logprob_hesapla(b_r_ids)

                loss, metrikler = self.loss_fn(
                    chosen_logps, rejected_logps, b_c_mask, b_r_mask
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                ep_loss += float(metrikler["kayip"].item())
                ep_cw += float(metrikler["chosen_odul"].item())
                ep_cr += float(metrikler["rejected_odul"].item())
                ep_df += float(metrikler["odul_farki"].item())
                ep_mv += float(metrikler["marjin_ihlali"].item())
                ep_acc += float(metrikler["dogruluk"].item())
                adim_sayisi += 1

            rapor["kayiplar"].append(ep_loss / adim_sayisi)
            rapor["chosen_odulleri"].append(ep_cw / adim_sayisi)
            rapor["rejected_odulleri"].append(ep_cr / adim_sayisi)
            rapor["odul_farklari"].append(ep_df / adim_sayisi)
            rapor["marjin_ihlalleri"].append((ep_mv / adim_sayisi) * 100.0)
            rapor["dogruluklar"].append((ep_acc / adim_sayisi) * 100.0)

        return rapor

    def mimari_4lu_kiyas_raporu(self) -> Dict[str, Any]:
        """PPO, DPO, ORPO ve SimPO arasındaki 4'lü mimari kıyaslama raporu."""
        return {
            "yontemler": ["PPO (RLHF)", "DPO", "ORPO", "SimPO"],
            "ref_model_gereksinimi": ["Evet (pi_ref)", "Evet (pi_ref)", "Hayır (0)", "Hayır (0)"],
            "uzunluk_normalizasyonu": ["Yok", "Opsiyonel", "Var", "Doğal Zorunlu (1/|y|)"],
            "hedef_marjin_gamma": ["Yok", "Yok", "Yok", "Var (γ = 0.5 - 1.4)"],
            "cikarim_uyumu": ["Orta", "Uyumsuz (P/P_ref)", "Uyumlu", "Mükemmel Uyumlu (P)"],
            "vram_kullanimi": ["%100 (4 Model)", "%50 (2 Model)", "%25 (1 Model)", "%25 (1 Model)"],
            "alpaca_eval_liderligi": ["Orta", "Yüksek", "Çok Yüksek", "Lider (SOTA)"],
        }
