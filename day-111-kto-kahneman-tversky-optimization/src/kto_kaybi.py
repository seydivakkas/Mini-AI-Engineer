"""
Kahneman-Tversky Optimization (KTO) Kayıp Fonksiyonu Modülü (Day 111).
Beklenti Teorisi (Prospect Theory), tekil ikili geri bildirimler (binary feedback)
ve asimetrik kayıp kaçınması (loss aversion) optimizasyonu.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class KTOLoss(nn.Module):
    """
    Kahneman-Tversky Optimization (KTO) Kayıp Fonksiyonu.
    L_KTO = lambda_D * (1 - sigma(r - z_ref)) [Eğer Beğenildiyse, y in D]
          + lambda_U * (1 - sigma(z_ref - r)) [Eğer Beğenilmediyse, y in U]
    """

    def __init__(
        self,
        beta: float = 0.1,
        lambda_d: float = 1.0,
        lambda_u: float = 1.33,  # Kayıp Kaçınması (Loss Aversion)
    ):
        super().__init__()
        self.beta = beta
        self.lambda_d = lambda_d
        self.lambda_u = lambda_u
        self.register_buffer("z_ref", torch.tensor(0.0))
        self.running_alpha = 0.05

    def forward(
        self,
        pi_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        labels: torch.Tensor,  # +1: Beğenildi (Desirable), -1: Beğenilmedi (Undesirable)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        KTO kaybını ve örtük ödülleri hesaplar.
        Girdiler:
          pi_logps: [B] Politika modelinin yanıt log-olasılıkları
          ref_logps: [B] Dondurulmuş referans modelin yanıt log-olasılıkları
          labels: [B] +1 (Desirable) veya -1 (Undesirable)
        Çıktı: (toplam_kayip, metrikler_sozlugu)
        """
        # 1. Örtük Ödül: r = beta * (log pi - log ref)
        r = self.beta * (pi_logps - ref_logps)

        # 2. Referans Noktası (z_ref) Güncellemesi
        batch_z_ref = r.detach().mean()
        if self.training:
            self.z_ref = (1.0 - self.running_alpha) * self.z_ref + self.running_alpha * batch_z_ref
        z_ref_val = self.z_ref if self.training else batch_z_ref

        # 3. Beklenti Teorisi Değer Fonksiyonu ve Asimetrik Kayıp
        mask_d = (labels > 0).float()  # Desirable (+1)
        mask_u = (labels < 0).float()  # Undesirable (-1)

        # Desirable Kaybı: 1 - sigmoid(r - z_ref)
        kayip_d = self.lambda_d * (1.0 - torch.sigmoid(r - z_ref_val))

        # Undesirable Kaybı: 1 - sigmoid(z_ref - r)
        kayip_u = self.lambda_u * (1.0 - torch.sigmoid(z_ref_val - r))

        toplam_ornek_d = mask_d.sum().clamp(min=1.0)
        toplam_ornek_u = mask_u.sum().clamp(min=1.0)

        kayip_d_ort = (kayip_d * mask_d).sum() / toplam_ornek_d
        kayip_u_ort = (kayip_u * mask_u).sum() / toplam_ornek_u

        toplam_kayip = kayip_d_ort + kayip_u_ort

        # Metrikler
        r_d = (r.detach() * mask_d).sum() / toplam_ornek_d
        r_u = (r.detach() * mask_u).sum() / toplam_ornek_u
        marjin = r_d - r_u

        # İkili Başarı Oranı (Desirable r > z_ref ve Undesirable r < z_ref)
        dogru_d = ((r.detach() > z_ref_val) * mask_d).sum() / toplam_ornek_d
        dogru_u = ((r.detach() < z_ref_val) * mask_u).sum() / toplam_ornek_u
        genel_dogruluk = 0.5 * (dogru_d + dogru_u)

        metrikler = {
            "toplam_kayip": toplam_kayip.detach(),
            "kayip_d": kayip_d_ort.detach(),
            "kayip_u": kayip_u_ort.detach(),
            "ortuk_odul_d": r_d,
            "ortuk_odul_u": r_u,
            "marjin": marjin,
            "dogruluk": genel_dogruluk,
            "z_ref": z_ref_val.detach(),
        }

        return toplam_kayip, metrikler


def hesapla_dizi_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    maske: torch.Tensor,
) -> torch.Tensor:
    """
    Logitlerden hedef token'ların toplam dizi log-olasılığını maskelenmiş olarak hesaplar.
    logits: [B, S, V]
    labels: [B, S]
    maske: [B, S]
    Çıktı: [B]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = maske[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return (per_token_logps * shift_mask).sum(dim=-1)
