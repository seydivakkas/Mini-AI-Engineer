"""
Direct Preference Optimization (DPO) Kayıp Fonksiyonu Modülü (Day 110).
Kapalı form log-oranı karşılaştırması, örtük ödül (implicit reward) ve marjin analizi.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Kayıp Fonksiyonu.
    L_DPO = -log sigma( beta * (log(pi(y_w)/ref(y_w)) - log(pi(y_l)/ref(y_l))) )
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        uzunluk_norm: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.uzunluk_norm = uzunluk_norm

    def forward(
        self,
        pi_logps_chosen: torch.Tensor,
        pi_logps_rejected: torch.Tensor,
        ref_logps_chosen: torch.Tensor,
        ref_logps_rejected: torch.Tensor,
        chosen_lens: Optional[torch.Tensor] = None,
        rejected_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        DPO kaybı ve örtük ödül metriklerini hesaplar.
        Girdiler: [B] boyutlu dizi log-olasılıkları.
        Çıktı: (kayip, metrikler_sozlugu)
        """
        if self.uzunluk_norm and chosen_lens is not None and rejected_lens is not None:
            pi_logps_chosen = pi_logps_chosen / chosen_lens.clamp(min=1)
            pi_logps_rejected = pi_logps_rejected / rejected_lens.clamp(min=1)
            ref_logps_chosen = ref_logps_chosen / chosen_lens.clamp(min=1)
            ref_logps_rejected = ref_logps_rejected / rejected_lens.clamp(min=1)

        # Log Oranları (Log-Ratios)
        pi_logratios = pi_logps_chosen - pi_logps_rejected
        ref_logratios = ref_logps_chosen - ref_logps_rejected

        logits = self.beta * (pi_logratios - ref_logratios)

        # Kayıp Hesabı (Label smoothing destekli)
        if self.label_smoothing > 0.0:
            kayip = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            kayip = -F.logsigmoid(logits).mean()

        # Örtük Ödüller (Implicit Rewards: r = beta * (log pi - log ref))
        ortuk_odul_chosen = self.beta * (pi_logps_chosen - ref_logps_chosen).detach()
        ortuk_odul_rejected = self.beta * (pi_logps_rejected - ref_logps_rejected).detach()
        ortuk_marjin = ortuk_odul_chosen - ortuk_odul_rejected

        # Sıralama Doğruluğu (% Accuracy)
        dogruluk = (logits > 0.0).float().mean()

        metrikler = {
            "kayip": kayip.detach(),
            "dogruluk": dogruluk.detach(),
            "ortuk_odul_chosen": ortuk_odul_chosen.mean(),
            "ortuk_odul_rejected": ortuk_odul_rejected.mean(),
            "ortuk_marjin": ortuk_marjin.mean(),
        }

        return kayip, metrikler


def hesapla_dizi_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    maske: torch.Tensor,
) -> torch.Tensor:
    """
    Logitlerden hedef token'ların toplam dizi log-olasılığını maskelenmiş olarak hesaplar.
    logits: [B, S, V]
    labels: [B, S]
    maske: [B, S] (Yalnızca yanıt kısımları 1, prompt ve padding 0)
    Çıktı: [B]
    """
    # Otoregresif kaydırma (Shift)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = maske[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Yalnızca geçerli yanıt token'larını topla
    dizi_logps = (per_token_logps * shift_mask).sum(dim=-1)
    return dizi_logps
