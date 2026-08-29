"""
Knowledge Distillation (Bilgi Damıtma) Kayıp Fonksiyonu Modülü (Day 119).
Hinton et al. yumuşatılmış logit sıcaklığı (T) ile KL Diverjansı ve Hard-Target Cross-Entropy birleşimi.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Öğretmen ve Öğrenci logitleri arasındaki bilgi aktarımını yöneten KD kaybı:
    L_Total = alpha * L_CE + (1 - alpha) * (T^2) * KL(P_T^T || P_S^T)
    """

    def __init__(
        self,
        sicaklik: float = 2.5,
        alpha: float = 0.3,
        pad_idx: int = -100,
    ):
        super().__init__()
        self.sicaklik = sicaklik
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(
        self,
        ogrenci_logits: torch.Tensor,
        ogretmen_logits: torch.Tensor,
        hedef_etiketler: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Girdiler:
          ogrenci_logits: [Batch, SeqLen, VocabSize]
          ogretmen_logits: [Batch, SeqLen, VocabSize]
          hedef_etiketler: [Batch, SeqLen]
        """
        B, S, V = ogrenci_logits.shape
        ogrenci_flat = ogrenci_logits.view(-1, V)
        ogretmen_flat = ogretmen_logits.view(-1, V)
        hedefler_flat = hedef_etiketler.view(-1)

        # 1. Sert Hedef (Hard-Target) Cross-Entropy Kaybı
        l_ce = self.ce_loss(ogrenci_flat, hedefler_flat)

        # 2. Yumuşatılmış Sıcaklık ile Log-Olasılık Dağılımları
        # log_softmax(S / T) ve log_softmax(T / T)
        s_log_prob = F.log_softmax(ogrenci_flat / self.sicaklik, dim=-1)
        t_log_prob = F.log_softmax(ogretmen_flat / self.sicaklik, dim=-1)

        # KL Divergence: KL(Öğretmen || Öğrenci)
        # PyTorch KLDivLoss(input, target) -> target * (log(target) - input)
        l_kl = self.kl_loss(s_log_prob, t_log_prob) * (self.sicaklik ** 2)

        # 3. Monolithic Toplam Kayıp
        l_total = self.alpha * l_ce + (1.0 - self.alpha) * l_kl

        metrikler = {
            "toplam_kayip": float(l_total.item()),
            "ce_kaybi": float(l_ce.item()),
            "kl_kaybi": float(l_kl.item()),
        }

        return l_total, metrikler
