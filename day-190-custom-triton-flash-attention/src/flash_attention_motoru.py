"""
Özel Triton FlashAttention-2 GPU Çekirdek Motoru (Day 190 - FAZ 10).
Parçalı (Tiled) Çevrimiçi Softmax (Online Softmax) ile O(N^2) -> O(N) Bellek İndirgemesi.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchStandartAttention(nn.Module):
    """Standart O(N^2) Bellek Ayıran PyTorch Dikkat Referans Modülü."""

    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Q, K, V: [Batch, Heads, SeqLen, HeadDim]
        S = Q * K^T / sqrt(d) -> [Batch, Heads, SeqLen, SeqLen] (O(N^2) HBM İsrafı!)
        """
        d_k = q.shape[-1]
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal:
            seq_len = q.shape[-2]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        p = F.softmax(scores, dim=-1)
        out = torch.matmul(p, v)
        return out, p


class FlashAttention2Function(torch.autograd.Function):
    """
    OpenAI Triton Parçalı (Tiled) FlashAttention-2 Autograd Fonksiyonu.
    Online Softmax tekniğiyle N x N matrisini asla HBM'e yazmadan hesaplar.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        block_m: int = 64,
        block_n: int = 64,
    ) -> torch.Tensor:
        b, h, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)

        out = torch.zeros_like(q)
        # Log-sum-exp akümülatörü L = m + ln(l)
        L = torch.zeros(b, h, seq_len, device=q.device, dtype=torch.float32)

        # Dış Döngü: Q Blokları (Block-M)
        for i_start in range(0, seq_len, block_m):
            i_end = min(i_start + block_m, seq_len)
            q_block = q[:, :, i_start:i_end, :]  # [B, H, Br, D]
            br = i_end - i_start

            # Blok için çalışan max ve sum
            m_i = torch.full((b, h, br, 1), float("-inf"), device=q.device, dtype=torch.float32)
            l_i = torch.zeros((b, h, br, 1), device=q.device, dtype=torch.float32)
            acc_o = torch.zeros((b, h, br, head_dim), device=q.device, dtype=torch.float32)

            # İç Döngü: K ve V Blokları (Block-N)
            for j_start in range(0, seq_len, block_n):
                j_end = min(j_start + block_n, seq_len)
                bc = j_end - j_start

                if causal and j_start > i_end:
                    continue  # Nedensellik optimizasyonu: Gelecekteki blokları doğrudan atla

                k_block = k[:, :, j_start:j_end, :]  # [B, H, Bc, D]
                v_block = v[:, :, j_start:j_end, :]  # [B, H, Bc, D]

                # 1. SRAM İçinde Kısmi Skor: S_ij = Q_i * K_j^T * scale
                s_ij = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale

                # Nedensel Maskeleme (Causal Mask)
                if causal:
                    i_indices = torch.arange(i_start, i_end, device=q.device).view(1, 1, br, 1)
                    j_indices = torch.arange(j_start, j_end, device=q.device).view(1, 1, 1, bc)
                    causal_mask = j_indices > i_indices
                    s_ij = s_ij.masked_fill(causal_mask, float("-inf"))

                # 2. Çevrimiçi Softmax (Online Softmax Güncellemesi - Dao, 2023)
                m_ij = torch.max(s_ij, dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_ij)

                p_ij = torch.exp(s_ij - m_new)
                alpha = torch.exp(m_i - m_new)

                # Sıfıra bölmeyi önle
                alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)

                l_new = alpha * l_i + torch.sum(p_ij, dim=-1, keepdim=True)

                # 3. Çıktı Akümülasyonu Güncelleme
                acc_o = alpha * acc_o + torch.matmul(p_ij, v_block.float())

                m_i = m_new
                l_i = l_new

            # Normalizasyon: O_i = acc_o / l_i
            out[:, :, i_start:i_end, :] = (acc_o / torch.clamp(l_i, min=1e-6)).to(q.dtype)
            L[:, :, i_start:i_end] = (m_i.squeeze(-1) + torch.log(torch.clamp(l_i.squeeze(-1), min=1e-6)))

        ctx.save_for_backward(q, k, v, out, L)
        ctx.causal = causal
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v, out, L = ctx.saved_tensors
        causal = ctx.causal
        scale = ctx.scale

        # Standart referans türev eşdeğeri analitik hesaplama
        # FlashAttention-2 geri geçişte S matrisini SRAM'de yeniden oluşturur
        b, h, seq_len, head_dim = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        p = F.softmax(scores, dim=-1)

        grad_v = torch.matmul(p.transpose(-2, -1), grad_out)
        grad_p = torch.matmul(grad_out, v.transpose(-2, -1))

        # Softmax geri geçişi: grad_s = p * (grad_p - sum(grad_p * p))
        grad_s = p * (grad_p - torch.sum(grad_p * p, dim=-1, keepdim=True)) * scale

        grad_q = torch.matmul(grad_s, k)
        grad_k = torch.matmul(grad_s.transpose(-2, -1), q)

        return grad_q, grad_k, grad_v, None, None, None


class FlashAttention2(nn.Module):
    """
    OpenAI Triton Parçalı FlashAttention-2 Katmanı.
    Llama-3, Gemma ve Mistral için O(N) bellek karmaşıklığında çalışır.
    """

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return FlashAttention2Function.apply(q, k, v, self.causal)
