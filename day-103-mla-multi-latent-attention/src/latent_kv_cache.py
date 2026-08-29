"""
DeepSeek Multi-Head Latent Attention (MLA) Sıkıştırılmış Önbellek Modülü (Day 103).
Tüm Key ve Value tensörlerini saklamak yerine sadece düşük dereceli ortak latent c^{KV} ve ayrık RoPE k^{R} saklar.
"""

from typing import Optional, Tuple
import torch


class LatentKVCache:
    """
    DeepSeek V2/V3 Multi-Head Latent Attention (MLA) Önbelleği.
    Klasik 2*H_kv*d_h tensörleri yerine yalnızca (d_c + d_R) boyutunda sıkıştırılmış latent saklar.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        kv_latent_dim: int = 512,
        rope_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.kv_latent_dim = kv_latent_dim
        self.rope_dim = rope_dim
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # 1. Sıkıştırılmış KV Latent Önbelleği: [B, max_seq_len, d_c]
        self.c_kv_cache = torch.zeros(
            (max_batch_size, max_seq_len, kv_latent_dim),
            dtype=dtype,
            device=self.device,
        )

        # 2. Ayrık RoPE Key Önbelleği: [B, max_seq_len, d_R]
        self.k_rope_cache = torch.zeros(
            (max_batch_size, max_seq_len, rope_dim),
            dtype=dtype,
            device=self.device,
        )

        self.mevcut_uzunluk = 0

    def guncelle(
        self,
        yeni_c_kv: torch.Tensor,
        yeni_k_rope: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Yeni gelen token'ların sıkıştırılmış latent'larını önbelleğe ekler.
        yeni_c_kv:   [B, S, d_c]
        yeni_k_rope: [B, S, d_R]
        """
        if self.c_kv_cache.dtype != yeni_c_kv.dtype:
            self.c_kv_cache = self.c_kv_cache.to(dtype=yeni_c_kv.dtype)
            self.k_rope_cache = self.k_rope_cache.to(dtype=yeni_k_rope.dtype)
            self.dtype = yeni_c_kv.dtype

        B, S, _ = yeni_c_kv.shape
        assert B <= self.max_batch_size, "Batch boyutu önbellek sınırını aştı."
        assert self.mevcut_uzunluk + S <= self.max_seq_len, "Dizi uzunluğu önbellek sınırını aştı."

        baslangic = self.mevcut_uzunluk
        bitis = baslangic + S

        self.c_kv_cache[:B, baslangic:bitis, :] = yeni_c_kv
        self.k_rope_cache[:B, baslangic:bitis, :] = yeni_k_rope
        self.mevcut_uzunluk = bitis

        aktif_c_kv = self.c_kv_cache[:B, :bitis, :]
        aktif_k_rope = self.k_rope_cache[:B, :bitis, :]
        return aktif_c_kv, aktif_k_rope

    def sifirla(self):
        """Önbelleği sıfırlar."""
        self.c_kv_cache.zero_()
        self.k_rope_cache.zero_()
        self.mevcut_uzunluk = 0

    def bellek_tuketimi_mb(self, katman_sayisi: int = 32) -> float:
        """
        MLA önbelleğinin belirli katman sayısında tükettiği toplam bellek (MB).
        Formül: Katman * Batch * SeqLen * (d_c + d_R) * Bayt/Eleman / (1024 * 1024)
        """
        eleman_basi_bayt = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        toplam_bayt = (
            katman_sayisi
            * self.max_batch_size
            * self.max_seq_len
            * (self.kv_latent_dim + self.rope_dim)
            * eleman_basi_bayt
        )
        return float(toplam_bayt / (1024 * 1024))
