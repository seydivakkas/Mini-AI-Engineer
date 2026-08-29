"""
Dairesel Halka Önbellek (Rolling Buffer Cache) Modülü (Day 105).
Mistral SWA mimarisi için sabit W boyutlu, modulo indeksleme (t mod W) ile çalışan önbellek.
"""

from typing import Optional, Tuple
import torch


class RollingBufferCache:
    """
    Mistral-7B Sliding Window Attention (SWA) Dairesel Halka Önbelleği.
    Dizi uzunluğu ne kadar uzarsa uzasın sadece son W adet Key ve Value saklar.
    Bellek karmaşıklığı: O(W) (Sabit / Constant Memory).
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        window_size: int = 512,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.max_batch_size = max_batch_size
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # Sabit W boyutlu Tensörler: [B, H_kv, W, head_dim]
        self.k_cache = torch.zeros(
            (max_batch_size, num_kv_heads, window_size, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (max_batch_size, num_kv_heads, window_size, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.toplam_eklenen_token = 0

    def guncelle(
        self,
        yeni_k: torch.Tensor,
        yeni_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Yeni gelen Key ve Value tensörlerini dairesel halka indeksine göre önbelleğe yazar.
        yeni_k, yeni_v: [B, H_kv, S, head_dim]
        """
        if self.k_cache.dtype != yeni_k.dtype:
            self.k_cache = self.k_cache.to(dtype=yeni_k.dtype)
            self.v_cache = self.v_cache.to(dtype=yeni_v.dtype)
            self.dtype = yeni_k.dtype

        B, H_kv, S, D = yeni_k.shape
        assert B <= self.max_batch_size, "Batch boyutu önbellek sınırını aştı."

        for i in range(S):
            slot = (self.toplam_eklenen_token + i) % self.window_size
            self.k_cache[:B, :, slot, :] = yeni_k[:, :, i, :]
            self.v_cache[:B, :, slot, :] = yeni_v[:, :, i, :]

        self.toplam_eklenen_token += S

        # Aktif önbellek dilimi
        if self.toplam_eklenen_token < self.window_size:
            aktif_k = self.k_cache[:B, :, :self.toplam_eklenen_token, :]
            aktif_v = self.v_cache[:B, :, :self.toplam_eklenen_token, :]
        else:
            aktif_k = self.k_cache[:B, :, :, :]
            aktif_v = self.v_cache[:B, :, :, :]

        return aktif_k, aktif_v

    def sifirla(self):
        """Önbelleği sıfırlar."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.toplam_eklenen_token = 0

    def bellek_tuketimi_mb(self, katman_sayisi: int = 32) -> float:
        """
        SWA önbelleğinin belirli katman sayısında tükettiği toplam sabit bellek (MB).
        Formül: 2 (K+V) * Katman * Batch * H_kv * WindowSize * HeadDim * Bayt/Eleman / (1024 * 1024)
        """
        eleman_basi_bayt = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        toplam_bayt = (
            2
            * katman_sayisi
            * self.max_batch_size
            * self.num_kv_heads
            * self.window_size
            * self.head_dim
            * eleman_basi_bayt
        )
        return float(toplam_bayt / (1024 * 1024))
