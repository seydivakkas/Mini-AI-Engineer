"""
4-Modelli PPO RLHF Hizalama Laboratuvarı (Day 109).
Actor, Critic, Frozen Reference Model ve Frozen Reward Model ile uçtan uca PPO eğitimi.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

from .ppo_matematigi import hesapla_kl_cezali_odul, hesapla_gae_avantaj, PPOClippedLoss
from .actor_critic_modelleri import ActorPolicy, CriticValueNetwork


class SahteOdulModeli(nn.Module):
    """PPO eğitimi için hedefli sentetik ödül fonksiyonu sağlayan model."""

    def __init__(self, hedef_mod: int = 4):
        super().__init__()
        self.hedef_mod = hedef_mod

    def forward(self, input_ids: torch.Tensor, yanit_baslangic: int) -> torch.Tensor:
        """Belirli yapısal desenlere (örn. çift/modlu tokenlar ve aralık) göre dinamik ödül üretir."""
        yanit = input_ids[:, yanit_baslangic:]  # [B, T]
        oran_mod = (yanit % self.hedef_mod == 0).float().mean(dim=-1)
        oran_aralik = ((yanit >= 200) & (yanit <= 700)).float().mean(dim=-1)
        # Ödül aralığı: [-1.5, +3.5]
        return (oran_mod * 2.5 + oran_aralik * 2.5) - 2.0


class PPOLaboratuvari:
    """4-Modelli PPO RLHF Hizalama Motoru."""

    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        cihaz: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.cihaz = cihaz

        # 1. Aktör (Eğitilebilir Politika)
        self.actor = ActorPolicy(vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers).to(cihaz)

        # 2. Eleştirmen (Eğitilebilir Değer Ağı)
        self.critic = CriticValueNetwork(vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers).to(cihaz)

        # 3. Referans Model (Dondurulmuş Başlangıç Politikası)
        self.ref_model = ActorPolicy(vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers).to(cihaz)
        self.ref_model.load_state_dict(self.actor.state_dict())
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # 4. Ödül Modeli (Dondurulmuş Kalite Puanlayıcı)
        self.reward_model = SahteOdulModeli(hedef_mod=4).to(cihaz)

        self.loss_fn = PPOClippedLoss(clip_eps=0.2, vf_coef=0.5)

        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=2e-3)

    def ppo_hizalama_adimi(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int = 12,
        kl_beta: float = 0.05,
        ppo_epochs: int = 4,
    ) -> Dict[str, float]:
        """Tek bir PPO Rollout + GAE + Optimizasyon adımını koşturur."""
        self.actor.eval()
        self.critic.eval()

        B, S_p = prompts.shape

        # 1. Rollout: Aktör metin üretir
        with torch.no_grad():
            tam_dizi, yanit_ids, old_logprobs = self.actor.uret_ve_logprob_al(
                prompts, max_new_tokens=max_new_tokens, temperature=0.9
            )
            # Referans model log-olasılıkları
            ref_logprobs = self.ref_model.logprob_degerlendir(tam_dizi, yanit_baslangic_idx=S_p)
            # Ödül modeli skaler ödülü
            rm_rewards = self.reward_model(tam_dizi, yanit_baslangic=S_p)
            # Eleştirmen başlangıç durum değerleri
            values_old = self.critic(tam_dizi, yanit_baslangic_idx=S_p)

            # 2. Token bazlı KL cezalı ödül ve GAE avantajları
            birlesik_odul, kl_div = hesapla_kl_cezali_odul(
                old_logprobs, ref_logprobs, rm_rewards, kl_beta=kl_beta
            )
            avantajlar, getiriler = hesapla_gae_avantaj(birlesik_odul, values_old)

        # 3. PPO Optimizasyon Adımı (Çoklu PPO Epoku)
        self.actor.train()
        self.critic.train()

        toplam_l, pol_l, val_l, clip_f = 0.0, 0.0, 0.0, 0.0
        for _ in range(ppo_epochs):
            new_logprobs = self.actor.logprob_degerlendir(tam_dizi, yanit_baslangic_idx=S_p)
            new_values = self.critic(tam_dizi, yanit_baslangic_idx=S_p)

            loss, pol_loss, val_loss, clip_frac = self.loss_fn(
                new_logprobs, old_logprobs, avantajlar, new_values, getiriler
            )

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.optimizer_actor.step()
            self.optimizer_critic.step()

            toplam_l += float(loss.item())
            pol_l += float(pol_loss.item())
            val_l += float(val_loss.item())
            clip_f += clip_frac

        return {
            "toplam_kayip": round(toplam_l / ppo_epochs, 4),
            "politika_kaybi": round(pol_l / ppo_epochs, 4),
            "deger_kaybi": round(val_l / ppo_epochs, 4),
            "kl_sapmasi": round(float(kl_div.item()), 4),
            "ortalama_odul": round(float(rm_rewards.mean().item()), 3),
            "kirpma_orani": round((clip_f / ppo_epochs) * 100.0, 2),
        }

    def ppo_egitim_dongusu(
        self,
        epok_sayisi: int = 15,
        batch_size: int = 32,
        prompt_len: int = 10,
        max_new_tokens: int = 12,
    ) -> Dict[str, List[float]]:
        """Çok adımlı PPO eğitim döngüsü koşturur ve gelişim eğrilerini kaydeder."""
        metrikler = {
            "oduller": [],
            "kl_sapmalari": [],
            "politika_kayiplari": [],
            "deger_kayiplari": [],
            "kirpma_oranlari": [],
        }

        for ep in range(epok_sayisi):
            # Rastgele promptlar
            prompts = torch.randint(1, 100, (batch_size, prompt_len), device=self.cihaz)
            adim_metrik = self.ppo_hizalama_adimi(prompts, max_new_tokens=max_new_tokens)

            metrikler["oduller"].append(adim_metrik["ortalama_odul"])
            metrikler["kl_sapmalari"].append(adim_metrik["kl_sapmasi"])
            metrikler["politika_kayiplari"].append(adim_metrik["politika_kaybi"])
            metrikler["deger_kayiplari"].append(adim_metrik["deger_kaybi"])
            metrikler["kirpma_oranlari"].append(adim_metrik["kirpma_orani"])

        return metrikler
