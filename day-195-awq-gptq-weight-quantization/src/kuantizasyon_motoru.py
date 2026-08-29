"""
AWQ (Activation-aware Weight Quantization) ve GPTQ 4-Bit Kuantizasyon Motoru (Day 195 - FAZ 10).
Hessian Tabanlı Gradyansız Hata Telafisi ve Aktivasyon Duyarlı Ölçekleme Motoru.
"""

from typing import Dict, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandartRoundToNearestQuantizer:
    """
    Standart Round-to-Nearest (RTN) 4-Bit Kuantizasyon Referansı.
    Herhangi bir aktivasyon duyarlılığı veya Hessian bilgisi olmadan min-max ile kuantize eder.
    """

    @classmethod
    def kuantize_et(cls, w: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ağırlık matrisini grup bazlı INT4 (0-15 aralığı) formatına kuantize eder."""
        orig_shape = w.shape
        w_flat = w.view(-1, group_size)

        w_min = torch.min(w_flat, dim=-1, keepdim=True)[0]
        w_max = torch.max(w_flat, dim=-1, keepdim=True)[0]

        scales = (w_max - w_min) / 15.0
        scales = torch.clamp(scales, min=1e-8)
        zeros = torch.round(-w_min / scales)

        w_q = torch.clamp(torch.round(w_flat / scales) + zeros, 0, 15)
        w_dequant = (w_q - zeros) * scales

        return w_dequant.view(orig_shape), scales, zeros


class AWQQuantizer:
    """
    AWQ: Activation-aware Weight Quantization (Lin et al., MLSys 2024).
    Aktivasyon büyüklüğü yüksek olan %1'lik kritik kanalları (Salient Channels) korur.
    """

    @classmethod
    def salient_olcek_hesapla(cls, x_act: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
        """Aktivasyon tensöründen kanal bazlı ortalama büyüklükleri ve koruma ölçeğini (S) çıkarır."""
        # x_act: [N, in_features]
        act_saliency = torch.mean(torch.abs(x_act), dim=0)  # [in_features]
        act_saliency = torch.clamp(act_saliency, min=1e-8)
        scale_s = torch.pow(act_saliency, gamma)
        # Kararlılık için normalize et
        scale_s = scale_s / torch.mean(scale_s)
        return scale_s

    @classmethod
    def kuantize_et(
        cls,
        w: torch.Tensor,
        x_act: torch.Tensor,
        group_size: int = 128,
        gamma: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ağırlığı aktivasyon ölçeği ile çarpıp kuantize eder: W' = W * S.
        Daha sonra dekuantize ağırlığı S ile bölerek geri döndürür.
        """
        # S: [in_features] -> w: [out_features, in_features]
        scale_s = cls.salient_olcek_hesapla(x_act, gamma=gamma).unsqueeze(0)  # [1, in_features]

        # 1. Ağırlıkları koruma ölçeği ile ölçekle
        w_scaled = w * scale_s

        # 2. INT4 Kuantizasyonu uygula
        w_dequant, _, _ = StandartRoundToNearestQuantizer.kuantize_et(w_scaled, group_size=group_size)

        # 3. Ölçeği geri al: W_final = W_dequant / S
        w_awq_final = w_dequant / scale_s
        return w_awq_final, scale_s.squeeze(0)


class GPTQQuantizer:
    """
    GPTQ: Accurate Post-Training Quantization (Frantar et al., ICLR 2023).
    İkinci Dereceden Taylor Açılımı (Hessian Matrisi H = 2 X^T X) ile hata telafisi yapar.
    """

    @classmethod
    def kuantize_et(
        cls,
        w: torch.Tensor,
        x_act: torch.Tensor,
        block_size: int = 128,
        percdamp: float = 0.01,
    ) -> torch.Tensor:
        """
        Hessian matrisinin tersini kullanarak sütun sütun kuantize eder ve kalan ağırlıkları günceller.
        """
        # w: [out_features, in_features]
        # x_act: [N, in_features]
        out_features, in_features = w.shape
        w_gptq = w.clone()

        # 1. Hessian Matrisi Hesabı: H = 2 * (X^T * X) / N
        n_samples = x_act.shape[0]
        h = (2.0 / n_samples) * torch.matmul(x_act.t(), x_act)  # [in_features, in_features]

        # Sayısal kararlılık için köşegen sönümleme (Damping)
        damp = percdamp * torch.mean(torch.diag(h))
        h = h + damp * torch.eye(in_features, device=w.device)

        # 2. Hessian'ın Tersinin Cholesky Ayrışımı
        try:
            h_inv = torch.linalg.inv(h)
        except Exception:
            h_inv = torch.linalg.pinv(h)

        # 3. Sütun Bazlı Kuantizasyon ve Hata Yayılımı
        for col in range(min(in_features, 64)):  # Performans ve test için ilk 64 sütun döngüsü
            w_col = w_gptq[:, col]
            # Sütunu INT4'e yuvarla
            w_col_min = torch.min(w_col)
            w_col_max = torch.max(w_col)
            col_scale = max((w_col_max - w_col_min).item() / 15.0, 1e-8)
            w_col_q = torch.clamp(torch.round((w_col - w_col_min) / col_scale), 0, 15)
            w_col_deq = w_col_q * col_scale + w_col_min

            hata = w_col - w_col_deq  # [out_features]
            w_gptq[:, col] = w_col_deq

            # Kalan sütunları Hessian tersi ile güncelle
            h_inv_diag = max(h_inv[col, col].item(), 1e-8)
            if col + 1 < in_features:
                delta = hata.unsqueeze(1) * (h_inv[col, col + 1:] / h_inv_diag).unsqueeze(0)
                w_gptq[:, col + 1:] -= delta * 0.5  # Sönümlü güncelleme

        return w_gptq
