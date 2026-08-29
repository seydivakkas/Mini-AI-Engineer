"""
KV Cache (Key-Value Önbelleği) Yönetim Modülü (Day 102).
Otoregresif üretim esnasında geçmiş token'ların Key ve Value tensörlerini saklar ve bellek tüketimini ölçer.
"""

from typing import Optional, Tuple
import torch


class KVCache:
    """
    Otoregresif LLM çıkarımında Key-Value tensörlerini saklayan ve güncelleyen önbellek.
    MHA, MQA ve GQA mimarilerinde bellek ayak izini hassas olarak takip eder.
    """
    def __init__(
        self,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # Önbellek Tensörleri: [B, H_kv, max_seq_len, head_dim]
        self.k_cache = torch.zeros(
            (max_batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (max_batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.mevcut_uzunluk = 0

    def guncelle(
        self,
        yeni_k: torch.Tensor,
        yeni_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Yeni gelen Key ve Value tensörlerini önbelleğe ekler ve o ana kadarki tüm K, V tensörlerini döner.
        yeni_k, yeni_v: [B, H_kv, seq_len, head_dim]
        """
        if self.k_cache.dtype != yeni_k.dtype:
            self.k_cache = self.k_cache.to(dtype=yeni_k.dtype)
            self.v_cache = self.v_cache.to(dtype=yeni_v.dtype)
            self.dtype = yeni_k.dtype
        B, H_kv, S, D = yeni_k.shape
        assert B <= self.max_batch_size, "Batch boyutu önbellek kapasitesini aştı."
        assert self.mevcut_uzunluk + S <= self.max_seq_len, "Dizi uzunluğu önbellek sınırını aştı."

        baslangic = self.mevcut_uzunluk
        bitis = baslangic + S

        self.k_cache[:B, :, baslangic:bitis, :] = yeni_k
        self.v_cache[:B, :, baslangic:bitis, :] = yeni_v
        self.mevcut_uzunluk = bitis

        aktif_k = self.k_cache[:B, :, :bitis, :]
        aktif_v = self.v_cache[:B, :, :bitis, :]
        return aktif_k, aktif_v

    def sifirla(self):
        """Önbelleği sıfırlar."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.mevcut_uzunluk = 0

    def bellek_tuketimi_mb(self, katman_sayisi: int = 32) -> float:
        """
        Belirli bir katman sayısı ve mevcut bağlam uzunluğuna göre toplam KV Cache bellek ayak izini (MB) hesaplar.
        Formül: 2 (K+V) * Katman * Batch * H_kv * SeqLen * HeadDim * Bayt/Eleman / (1024 * 1024)
        """
        eleman_basi_bayt = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        toplam_bayt = (
            2
            * katman_sayisi
            * self.max_batch_size
            * self.num_kv_heads
            * self.max_seq_len
            * self.head_dim
            * eleman_basi_bayt
        )
        return float(toplam_bayt / (1024 * 1024))
