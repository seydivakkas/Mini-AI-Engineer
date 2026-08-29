"""
Görsel SFT Kayıp Maskeleme (Visual Loss Masking) Modülü (Day 163 - FAZ 9).
Görüntü patch tokenlarını (256) ve kullanıcı prompt tokenlarını -100 ile maskeleyerek yalnızca asistan yanıtı üzerinde kayıp hesaplar.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualLossMaskeleyici:
    """Görsel ve Komut Tokenlarını Yoksayan (-100 Masking) Kayıp Motoru."""

    @classmethod
    def hedef_maskeli_etiket_olustur(
        cls,
        visual_token_count: int,
        prompt_token_ids: torch.Tensor,
        response_token_ids: torch.Tensor,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Girdi Tokenları: [Visual Tokens (256) + Prompt Tokens (L_p) + Response Tokens (L_r)]
        Hedef Etiketler: [-100 (256 adet)     + -100 (L_p adet)     + Response IDs (L_r adet)]
        """
        B = prompt_token_ids.shape[0]
        L_p = prompt_token_ids.shape[1]
        L_r = response_token_ids.shape[1]

        # 1. Girdi Metin Tokenları
        input_text_ids = torch.cat([prompt_token_ids, response_token_ids], dim=1)  # (B, L_p + L_r)

        # 2. Toplam Hedef Etiket Tensörü (Görsel + Metin Boyutunda)
        toplam_uzunluk = visual_token_count + L_p + L_r
        labels = torch.full((B, toplam_uzunluk), ignore_index, dtype=torch.long, device=prompt_token_ids.device)

        # 3. Yalnızca Asistan Yanıtı bölgesine gerçek token ID'lerini yerleştir
        asistan_baslangic = visual_token_count + L_p
        labels[:, asistan_baslangic:] = response_token_ids

        return input_text_ids, labels

    @classmethod
    def maskeli_cross_entropy_kaybi_hesapla(
        cls,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Logits: (Batch, Seq_Len, Vocab_Size)
        Labels: (Batch, Seq_Len) -> Görsel ve Prompt pozisyonları -100
        """
        # Next-token prediction için 1 adım kaydırma
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )
        return loss
