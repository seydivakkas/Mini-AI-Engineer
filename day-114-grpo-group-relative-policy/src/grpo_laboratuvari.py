"""
GRPO Akıl Yürütme (Reasoning) Laboratuvarı ve PPO vs GRPO Kıyas Motoru (Day 114).
Kural tabanlı doğruluk ve düşünme (<think>) ödülü, G=8 grup örneklemesi ve DeepSeek-R1 tarzı eğitim.
"""

from typing import Dict, Any, List, Tuple
import copy
import torch
import torch.nn as nn

from .grpo_kaybi import GRPOLoss, grup_goreli_avantaj_hesapla
from .grpo_modeli import GRPODilModeli


class GRPOLaboratuvari:
    """GRPO Akıl Yürütme ve Grup Tercih Laboratuvarı."""

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

        # 1. Politika Modeli (Eğitilebilir)
        self.model = GRPODilModeli(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers
        ).to(cihaz)

        # 2. Dondurulmuş Referans Modeli (KL cezası için)
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Critic (Value Network) YOK! (0 Parametre)
        self.loss_fn = GRPOLoss(clip_eps=0.2, beta_kl=0.04)

    def kural_tabanli_odul_fonksiyonu(
        self,
        uretilen_diziler: torch.Tensor,
        prompt_len: int,
        hedef_aralik_min: int,
        hedef_aralik_max: int,
    ) -> torch.Tensor:
        """
        DeepSeek-R1 tarzı kural tabanlı ödül motoru:
        - Doğruluk Ödülü (+1.0): Üretilen tokenlar hedef token aralığında mı?
        - Düşünme / Akıl Yürütme Format Ödülü (+0.5): Çözüm adımı tokenları var mı?
        """
        G = uretilen_diziler.shape[0]
        yanitlar = uretilen_diziler[:, prompt_len:]  # [G, S_resp]

        oduller = torch.zeros(G, device=self.cihaz, dtype=torch.float32)

        for i in range(G):
            yanit_tokenlari = yanitlar[i]
            # 1. Doğruluk kontrolü: hedef aralıktaki token sayısı
            dogru_token_sayisi = ((yanit_tokenlari >= hedef_aralik_min) & (yanit_tokenlari < hedef_aralik_max)).sum().item()
            dogruluk_skoru = min(1.0, dogru_token_sayisi / 4.0)

            # 2. Format kontrolü: Yanıt uzunluğu ve çeşitlilik
            benzersiz_token = len(torch.unique(yanit_tokenlari))
            format_skoru = 0.5 if benzersiz_token >= 3 else 0.1

            oduller[i] = dogruluk_skoru + format_skoru

        return oduller  # [G]

    def grpo_egit(
        self,
        prompt_sayisi: int = 50,
        group_size: int = 8,
        epok_sayisi: int = 15,
        lr: float = 5e-4,
    ) -> Dict[str, List[float]]:
        """GRPO grup örneklemeli akıl yürütme eğitimi."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # Sentetik Akıl Yürütme Promptları [N, 8]
        prompt_len = 8
        resp_len = 12
        prompts = torch.randint(1, int(self.vocab_size * 0.2), (prompt_sayisi, prompt_len), device=self.cihaz)

        hedef_min = int(self.vocab_size * 0.4)
        hedef_max = int(self.vocab_size * 0.7)

        rapor = {
            "toplam_kayiplar": [],
            "politika_kayiplari": [],
            "kl_kayiplari": [],
            "ortalama_oduller": [],
            "std_oduller": [],
            "kirpilma_oranlari": [],
        }

        for ep in range(epok_sayisi):
            self.model.train()
            ep_tot, ep_pol, ep_kl, ep_rew, ep_std, ep_clip = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            for p_idx in range(prompt_sayisi):
                tek_prompt = prompts[p_idx : p_idx + 1]  # [1, 8]

                # 1. G Adet Çıktı Örnekle (Group Rollout)
                with torch.no_grad():
                    grup_diziler = self.model.grup_ornekle(
                        tek_prompt, group_size=group_size, max_new_tokens=resp_len, temperature=1.0
                    )  # [G, 20]
                    logp_old = self.model.token_logprob_hesapla(grup_diziler)
                    logp_ref = self.ref_model.token_logprob_hesapla(grup_diziler)

                    # Kural tabanlı ödülleri hesapla: [G]
                    oduller = self.kural_tabanli_odul_fonksiyonu(
                        grup_diziler, prompt_len, hedef_min, hedef_max
                    )

                # Maske (yalnızca yanıt tokenları)
                token_mask = torch.zeros_like(grup_diziler, dtype=torch.float32)
                token_mask[:, prompt_len:] = 1.0

                # 2. Politika Modeli İleri Geçişi
                logp_theta = self.model.token_logprob_hesapla(grup_diziler)

                loss, metrikler = self.loss_fn(
                    logp_theta=logp_theta,
                    logp_old=logp_old,
                    logp_ref=logp_ref,
                    oduller=oduller,
                    token_mask=token_mask,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                ep_tot += float(metrikler["toplam_kayip"].item())
                ep_pol += float(metrikler["politika_kaybi"].item())
                ep_kl += float(metrikler["kl_kaybi"].item())
                ep_rew += float(metrikler["ortalama_odul"].item())
                ep_std += float(metrikler["std_odul"].item())
                ep_clip += float(metrikler["kirpilma_orani"].item())

            rapor["toplam_kayiplar"].append(ep_tot / prompt_sayisi)
            rapor["politika_kayiplari"].append(ep_pol / prompt_sayisi)
            rapor["kl_kayiplari"].append(ep_kl / prompt_sayisi)
            rapor["ortalama_oduller"].append(ep_rew / prompt_sayisi)
            rapor["std_oduller"].append(ep_std / prompt_sayisi)
            rapor["kirpilma_oranlari"].append((ep_clip / prompt_sayisi) * 100.0)

        return rapor

    def ppo_vs_grpo_kiyas_raporu(self) -> Dict[str, Any]:
        """PPO ve GRPO arasındaki mimari ve kaynak kıyası."""
        return {
            "kriterler": ["Critic Modeli (V)", "Avantaj Tahmini", "Grup Örnekleme (G)", "VRAM Kullanımı", "Akıl Yürütme Uyumu"],
            "ppo": ["Var (70B Parametre)", "GAE-lambda (Değer Hatası Var)", "1-2 Rollout", "%100 (4 Model)", "Düşük/Orta"],
            "grpo": ["YOK (0 Parametre)", "Grup Z-Score ((r-mean)/std)", "G=8, 64, 128 Rollout", "%35 (Policy+Ref)", "Çok Yüksek (DeepSeek-R1)"],
        }
