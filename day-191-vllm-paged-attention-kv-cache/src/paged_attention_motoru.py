"""
vLLM PagedAttention ve Dinamik KV Cache Yönetim Motoru (Day 191 - FAZ 10).
Sanal Bellek Sayfalama Mimarisi, Blok Tablosu ve Copy-on-Write (CoW) Mekanizması.
"""

from typing import List, Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FizikselBlokYonetici:
    """
    GPU VRAM Fiziksel Blok Yöneticisi (Block Allocator).
    Sabit boyutlu (Block Size: 16 Token) bellek blok havuzunu yönetir.
    """

    def __init__(self, toplam_blok_sayisi: int = 512, blok_boyutu: int = 16):
        self.toplam_blok_sayisi = toplam_blok_sayisi
        self.blok_boyutu = blok_boyutu
        self.bostaki_bloklar: List[int] = list(range(toplam_blok_sayisi))
        self.referans_sayaclari: Dict[int, int] = {b: 0 for b in range(toplam_blok_sayisi)}

    def blok_tahsis_et(self) -> int:
        """Boştaki blok havuzundan yeni bir fiziksel blok ayırır."""
        if not self.bostaki_bloklar:
            raise MemoryError("GPU VRAM KV Cache Doldu! Boş fiziksel blok kalmadı.")
        blok_id = self.bostaki_bloklar.pop(0)
        self.referans_sayaclari[blok_id] = 1
        return blok_id

    def blok_serbest_birak(self, blok_id: int):
        """Blok referans sayacını düşürür; sıfır olunca havuza iade eder."""
        if self.referans_sayaclari[blok_id] > 0:
            self.referans_sayaclari[blok_id] -= 1
            if self.referans_sayaclari[blok_id] == 0:
                self.bostaki_bloklar.append(blok_id)

    def referans_arttir(self, blok_id: int):
        """Paralel örnekleme veya Beam Search için bloğu paylaşır (Ref Count +1)."""
        self.referans_sayaclari[blok_id] += 1

    def bos_blok_orani(self) -> float:
        """Kullanılabilir bellek oranını döndürür."""
        return len(self.bostaki_bloklar) / float(self.toplam_blok_sayisi)


class GelenIstek:
    """
    LLM Çıkarım İsteği ve Mantıksal-Fiziksel Blok Eşleme Tablosu.
    """

    def __init__(self, istek_id: str, prompt_token_sayisi: int = 0):
        self.istek_id = istek_id
        self.prompt_len = prompt_token_sayisi
        self.toplam_token_sayisi = 0
        self.blok_tablosu: List[int] = []  # Mantıksal Sayfa -> Fiziksel Blok ID


class PagedKVCache:
    """
    Parçalanmamış (Non-Contiguous) Fiziksel KV Cache Tensör Deposu.
    """

    def __init__(
        self,
        toplam_blok_sayisi: int = 512,
        blok_boyutu: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        self.blok_boyutu = blok_boyutu
        self.num_heads = num_heads
        self.head_dim = head_dim

        # [NumBlocks, NumHeads, BlockSize, HeadDim]
        self.k_cache = torch.zeros(toplam_blok_sayisi, num_heads, blok_boyutu, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(toplam_blok_sayisi, num_heads, blok_boyutu, head_dim, dtype=dtype)

    def token_kv_yaz(
        self,
        istek: GelenIstek,
        k_vektor: torch.Tensor,
        v_vektor: torch.Tensor,
        blok_yoneticisi: FizikselBlokYonetici,
    ):
        """
        Yeni üretilen token için KV vektörünü ilgili fiziksel bloğa yazar.
        Gerektiğinde dinamik yeni blok tahsis eder ve token sayacını arttırır.
        """
        token_sirasi = istek.toplam_token_sayisi
        mantiksal_blok_indeksi = token_sirasi // self.blok_boyutu
        blok_ici_ofset = token_sirasi % self.blok_boyutu

        # Yeni bloğa ihtiyaç var mı?
        while mantiksal_blok_indeksi >= len(istek.blok_tablosu):
            yeni_blok_id = blok_yoneticisi.blok_tahsis_et()
            istek.blok_tablosu.append(yeni_blok_id)

        fiziksel_blok_id = istek.blok_tablosu[mantiksal_blok_indeksi]

        # Fiziksel tensör alanına yaz
        self.k_cache[fiziksel_blok_id, :, blok_ici_ofset, :] = k_vektor
        self.v_cache[fiziksel_blok_id, :, blok_ici_ofset, :] = v_vektor
        istek.toplam_token_sayisi += 1


class PagedAttentionEngine:
    """
    vLLM PagedAttention Kod Çözme (Decoding) Yürütücüsü.
    Non-contiguous fiziksel bloklardan KV vektörlerini toplayarak dikkat hesaplar.
    """

    def __init__(self, kv_cache: PagedKVCache, blok_yoneticisi: FizikselBlokYonetici):
        self.kv_cache = kv_cache
        self.blok_yoneticisi = blok_yoneticisi
        self.scale = 1.0 / math.sqrt(kv_cache.head_dim)

    def tek_token_dikkat(
        self,
        istek: GelenIstek,
        q_token: torch.Tensor,  # [NumHeads, HeadDim]
    ) -> torch.Tensor:
        """
        Gelen sorgu tokenı (q) ile isteğin geçmişteki tüm tokenları arasında dikkat hesaplar.
        """
        num_tokens = istek.toplam_token_sayisi
        if num_tokens == 0:
            raise ValueError("İstekte kayıtlı KV tokenı bulunmamaktadır.")

        # 1. İstek için tahsis edilmiş fiziksel bloklardan K ve V toplanır
        k_toplu = []
        v_toplu = []

        for b_idx, blok_id in enumerate(istek.blok_tablosu):
            gecerli_uzunluk = min(self.kv_cache.blok_boyutu, num_tokens - (b_idx * self.kv_cache.blok_boyutu))
            if gecerli_uzunluk <= 0:
                break
            k_toplu.append(self.kv_cache.k_cache[blok_id, :, :gecerli_uzunluk, :])
            v_toplu.append(self.kv_cache.v_cache[blok_id, :, :gecerli_uzunluk, :])

        k_gecmis = torch.cat(k_toplu, dim=1)  # [NumHeads, NumTokens, HeadDim]
        v_gecmis = torch.cat(v_toplu, dim=1)  # [NumHeads, NumTokens, HeadDim]

        # 2. Dikkat Skoru Hesapla: S = (q * K^T) * scale
        q_genisletilmis = q_token.unsqueeze(1)  # [NumHeads, 1, HeadDim]
        skorlar = torch.matmul(q_genisletilmis, k_gecmis.transpose(-2, -1)) * self.scale  # [NumHeads, 1, NumTokens]

        p = F.softmax(skorlar, dim=-1)  # [NumHeads, 1, NumTokens]
        cikti = torch.matmul(p, v_gecmis).squeeze(1)  # [NumHeads, HeadDim]

        return cikti
