"""
GRPO (Group Relative Policy Optimization) Motoru (Day 202 - FAZ 11).
DeepSeekMath & DeepSeek-R1 Tarzı Değer Modeli (Critic) Olmayan Grup İçi Bağıl Avantaj ile Akıl Yürütme Eğitimi.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import re


class MathProblemEnvironment:
    """
    Deterministik Matematik Problemi Üreteci ve Doğrulama Ortamı.
    """

    @staticmethod
    def rastgele_problem_uret() -> Dict[str, Any]:
        """Çözümü kesin olan cebirsel veya aritmetik soru üretir."""
        tur = random.choice(["denklem", "moduler", "kombinasyon"])
        if tur == "denklem":
            # ax + b = c
            a = random.randint(2, 6)
            x_ans = random.randint(3, 15)
            b = random.randint(1, 20)
            c = a * x_ans + b
            soru = f"{a}x + {b} = {c} denkleminde x değerini bulunuz."
            return {"soru": soru, "tur": tur, "dogru_cevap": str(x_ans)}
        elif tur == "moduler":
            # (a * b) mod m
            a = random.randint(12, 35)
            b = random.randint(4, 15)
            m = random.randint(5, 11)
            ans = (a * b) % m
            soru = f"({a} * {b}) mod {m} işleminin sonucunu bulunuz."
            return {"soru": soru, "tur": tur, "dogru_cevap": str(ans)}
        else:
            # n! / (n-2)! = n*(n-1)
            n = random.randint(4, 10)
            ans = n * (n - 1)
            soru = f"{n}! / {n-2}! faktöriyel sadeleştirmesinin sonucunu bulunuz."
            return {"soru": soru, "tur": tur, "dogru_cevap": str(ans)}


class RuleBasedMathRewardVerifier:
    """
    DeepSeek-R1 Kural Tabanlı Çoklu Ödül Doğrulayıcısı (Rule-Based Verifier).
    Format Ödülü (Biçim) + Doğruluk Ödülü (Accuracy).
    """

    @staticmethod
    def odul_hesapla(yanit_metni: str, beklenen_cevap: str) -> Dict[str, float]:
        """Biçim (<think>...</think>) ve kesin sayısal eşleşmeyi ödüllendirir."""
        # 1. Format Ödülü (Biçimsel Düşünce Etiketleri)
        has_think_start = "<think>" in yanit_metni
        has_think_end = "</think>" in yanit_metni
        format_odulu = 0.2 if (has_think_start and has_think_end) else 0.0

        # 2. Doğruluk Ödülü (Exact Match Accuracy)
        # </think> etiketinden sonraki nihai cevap kısmını tara
        if has_think_end:
            nihai_kisim = yanit_metni.split("</think>")[-1]
        else:
            nihai_kisim = yanit_metni

        # Metindeki sayıları çıkar
        sayilar = re.findall(r"\b\d+\b", nihai_kisim)
        dogruluk_odulu = 0.0
        if sayilar and beklenen_cevap in sayilar:
            dogruluk_odulu = 0.8  # Tam puan

        toplam_odul = format_odulu + dogruluk_odulu
        return {
            "format_odulu": format_odulu,
            "dogruluk_odulu": dogruluk_odulu,
            "toplam_odul": toplam_odul,
        }


class PolicyModel(nn.Module):
    """Küçük Parametreli Politika Modeli (Policy Network)."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self.embedding(x)))
        return self.head(h)


class GRPOTrainer:
    """
    DeepSeek-R1 GRPO (Group Relative Policy Optimization) Eğitim Motoru.
    Ayrı Değer Modeli (Critic Network) OLMADAN grup içi bağıl avantajı hesaplar.
    """

    def __init__(
        self,
        policy_model: Optional[PolicyModel] = None,
        group_size: int = 4,
        clip_eps: float = 0.2,
        beta_kl: float = 0.04,
        lr: float = 1e-3,
    ):
        self.policy = policy_model if policy_model is not None else PolicyModel()
        self.ref_policy = PolicyModel()
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()

        self.group_size = group_size
        self.clip_eps = clip_eps
        self.beta_kl = beta_kl
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def grup_ici_bagil_avantaj_hesapla(self, oduller: List[float]) -> torch.Tensor:
        """
        A_i = (r_i - mean(R)) / (std(R) + eps)
        Ödüllerin grup içi standardizasyonu (Değer Modeline İhtiyaç Yoktur!).
        """
        r_tensor = torch.tensor(oduller, dtype=torch.float32)
        mean_r = r_tensor.mean()
        std_r = r_tensor.std()

        if std_r < 1e-6:
            # Gruptaki tüm yanıtlar aynı ödülü aldıysa avantaj 0'dır
            return torch.zeros_like(r_tensor)

        advantage = (r_tensor - mean_r) / (std_r + 1e-4)
        return advantage

    def ornek_grup_ciktilari_uret(self, soru_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Grup boyutu (G=4) kadar aday düşünce ve yanıt zinciri üretir."""
        grup_ciktilar = []
        dogru = soru_dict["dogru_cevap"]

        for i in range(self.group_size):
            # Model yetkinliğine göre doğru/yanlış üretme simülasyonu
            is_correct = random.random() < 0.60
            has_format = random.random() < 0.85

            if is_correct and has_format:
                yanit = (
                    f"<think>\n1. Verilen soru: {soru_dict['soru']}\n"
                    f"2. Adım adım denklem çözüldü.\n"
                    f"3. Sonuç doğrulandı.\n</think>\nCevap: {dogru}"
                )
            elif not is_correct and has_format:
                yanlis_sayi = str(int(dogru) + random.choice([-2, -1, 1, 3]))
                yanit = (
                    f"<think>\n1. Hatalı işlem yapıldı.\n</think>\nCevap: {yanlis_sayi}"
                )
            else:
                yanlis_sayi = str(int(dogru) + 5)
                yanit = f"Doğrudan cevap: {yanlis_sayi}"

            odul_raporu = RuleBasedMathRewardVerifier.odul_hesapla(yanit, dogru)
            grup_ciktilar.append({
                "aday_idx": i,
                "yanit": yanit,
                "odul": odul_raporu["toplam_odul"],
                "dogruluk_odulu": odul_raporu["dogruluk_odulu"],
                "format_odulu": odul_raporu["format_odulu"],
            })

        return grup_ciktilar

    def grpo_egitim_adimi(self, soru_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Tek bir GRPO güncelleme adımı yürütür."""
        grup_ciktilar = self.ornek_grup_ciktilari_uret(soru_dict)
        oduller = [c["odul"] for c in grup_ciktilar]
        avantajlar = self.grup_ici_bagil_avantaj_hesapla(oduller)

        # Sahte tensör üzerinde GRPO Loss hesabı
        self.optimizer.zero_grad()
        dummy_input = torch.randint(0, 128, (self.group_size, 10))

        logits = self.policy(dummy_input)
        with torch.no_grad():
            ref_logits = self.ref_policy(dummy_input)

        log_probs = F.log_softmax(logits, dim=-1).mean(dim=[1, 2])
        ref_log_probs = F.log_softmax(ref_logits, dim=-1).mean(dim=[1, 2])

        # Politika Oranı: rho = exp(log_p - ref_log_p)
        ratio = torch.exp(log_probs - ref_log_probs)

        # Kırpılmış (Clipped) GRPO Kaybı
        surr1 = ratio * avantajlar
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * avantajlar
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL Cezası (Referans Modelden Fazla Uzaklaşmama)
        kl_div = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction="batchmean",
        )
        total_loss = policy_loss + self.beta_kl * kl_div

        total_loss.backward()
        self.optimizer.step()

        return {
            "toplam_kayip": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl_div": float(kl_div.item()),
            "ortalama_odul": float(np.mean(oduller)),
            "avantaj_standart_sapma": float(avantajlar.std().item()) if len(avantajlar) > 1 else 0.0,
            "grup_ciktilari": grup_ciktilar,
        }
