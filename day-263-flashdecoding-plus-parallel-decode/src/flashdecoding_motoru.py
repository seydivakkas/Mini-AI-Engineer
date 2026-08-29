"""
FlashDecoding++: Devasa Batch Boyutlarında KV-Cache Bölümleme ile Decode Hızlandırma Motoru (Day 263).
Split-K Attention, Dynamic Softmax Rescaling ve Tree-Reduction.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class KVCacheManager:
    """Paged / Chunked KV-Cache Bellek Yöneticisi."""

    def __init__(self, batch_size: int, n_heads: int, head_dim: int, max_seq_len: int = 32768):
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.k_cache = np.zeros((batch_size, n_heads, max_seq_len, head_dim), dtype=np.float32)
        self.v_cache = np.zeros((batch_size, n_heads, max_seq_len, head_dim), dtype=np.float32)
        self.current_seq_len = 0

    def append(self, k_step: np.ndarray, v_step: np.ndarray):
        """Yeni üretilen tek adımlık K ve V vektörlerini önbelleğe ekler."""
        assert self.current_seq_len < self.max_seq_len, "KV-Cache bellek sınırına ulaşıldı!"
        self.k_cache[:, :, self.current_seq_len, :] = k_step
        self.v_cache[:, :, self.current_seq_len, :] = v_step
        self.current_seq_len += 1

    def get_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dolu olan aktif KV-Cache dilimini döndürür."""
        return (
            self.k_cache[:, :, : self.current_seq_len, :],
            self.v_cache[:, :, : self.current_seq_len, :],
        )


class FlashDecodingPlusEngine:
    """FlashDecoding++: Split-K Paralel KV-Cache ve Softmax Rescaling Çekirdeği."""

    @classmethod
    def execute_split_k(
        cls,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        chunk_size: int = 256,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Split-K algoritmasıyla uzun bağlam KV-Cache Attention hesaplar.
        q: [B, H, 1, D]
        k: [B, H, S, D]
        v: [B, H, S, D]
        """
        b, h, q_len, d = q.shape
        _, _, s, _ = k.shape
        scale = 1.0 / math.sqrt(d)

        num_chunks = max(1, math.ceil(s / chunk_size))

        partial_outputs = []
        partial_maxes = []
        partial_sums = []

        # 1. PARALEL SPLIT-K AŞAMASI (Her chunk ayrı bir GPU SM'de koşar)
        for c in range(num_chunks):
            start_idx = c * chunk_size
            end_idx = min((c + 1) * chunk_size, s)

            k_chunk = k[:, :, start_idx:end_idx, :]  # [B, H, C_len, D]
            v_chunk = v[:, :, start_idx:end_idx, :]  # [B, H, C_len, D]

            # S_c = (Q @ K_c^T) * scale -> [B, H, 1, C_len]
            s_c = np.matmul(q, k_chunk.swapaxes(-1, -2)) * scale

            # Yerel Max (m_c)
            m_c = np.max(s_c, axis=-1, keepdims=True)  # [B, H, 1, 1]

            # P_c = exp(S_c - m_c)
            p_c = np.exp(s_c - m_c)

            # Yerel Sum (l_c)
            l_c = np.sum(p_c, axis=-1, keepdims=True)  # [B, H, 1, 1]

            # O_c = P_c @ V_c -> [B, H, 1, D] (Normalize edilmemiş ağırlıklı toplam)
            o_c = np.matmul(p_c, v_chunk)

            partial_outputs.append(o_c)
            partial_maxes.append(m_c)
            partial_sums.append(l_c)

        # 2. HİYERARŞİK TREE-REDUCTION VE SOFTMAX RESCALING AŞAMASI
        # Global Max: m = max(m_0, m_1, ..., m_{K-1})
        all_maxes = np.concatenate(partial_maxes, axis=-1)  # [B, H, 1, num_chunks]
        global_max = np.max(all_maxes, axis=-1, keepdims=True)  # [B, H, 1, 1]

        # Yeniden ölçekleme katsayıları: alpha_c = exp(m_c - global_max)
        rescaled_outputs = []
        rescaled_sums = []

        for c in range(num_chunks):
            alpha_c = np.exp(partial_maxes[c] - global_max)  # [B, H, 1, 1]
            rescaled_o_c = partial_outputs[c] * alpha_c     # [B, H, 1, D]
            rescaled_l_c = partial_sums[c] * alpha_c        # [B, H, 1, 1]

            rescaled_outputs.append(rescaled_o_c)
            rescaled_sums.append(rescaled_l_c)

        # Global toplam bölen (Denominator L)
        total_sum = np.sum(np.concatenate(rescaled_sums, axis=-1), axis=-1, keepdims=True)  # [B, H, 1, 1]
        sum_outputs = np.sum(np.stack(rescaled_outputs, axis=0), axis=0)                     # [B, H, 1, D]

        # Nihai çıktı O = sum(alpha_c * O_c) / L
        final_output = sum_outputs / (total_sum + 1e-8)

        stats = {
            "toplam_baglam_uzunlugu": s,
            "bolumlenen_chunk_sayisi": num_chunks,
            "chunk_boyutu": chunk_size,
            "sm_paralellik_artisi": f"{num_chunks}x SM Paralelizasyonu",
        }

        return final_output, stats
