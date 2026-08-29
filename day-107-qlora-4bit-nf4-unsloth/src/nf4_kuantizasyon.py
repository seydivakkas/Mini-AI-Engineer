"""
NormalFloat4 (NF4) ve Double Quantization (DQ) Kuantizasyon Modülü (Day 107).
QLoRA (Dettmers et al., 2023) teorik 16-noktalı normal dağılım kuantizasyonu ve ikincil ölçekleme.
"""

from typing import Tuple
import torch

# QLoRA Makalesi (Dettmers et al., 2023) Standart Normal Dağılım Ters CDF 16-Nokta NF4 Kuantile Tablosu
NF4_SEVIYELER = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


class NF4Kuantizator:
    """4-bit NormalFloat (NF4) Blok Bazlı Kuantizasyon ve Dekuantizasyon Motoru."""

    def __init__(self, block_size: int = 64, device: torch.device = torch.device("cpu")):
        self.block_size = block_size
        self.device = device
        self.nf4_lut = torch.tensor(NF4_SEVIYELER, dtype=torch.float32, device=device)

    def kuantize_et(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """
        Girdi tensörünü blok bazlı (B=64) 4-bit NF4 indekslerine dönüştürür.
        Çıktı: (q_indeksler [N], c1_skalalar [N/B], orijinal_sekil)
        """
        orijinal_sekil = tensor.shape
        t_duz = tensor.flatten().to(dtype=torch.float32, device=self.device)
        N = t_duz.numel()

        # Blok boyutuna hizalama (Padding)
        pad_miktari = (self.block_size - (N % self.block_size)) % self.block_size
        if pad_miktari > 0:
            t_duz = torch.nn.functional.pad(t_duz, (0, pad_miktari), value=0.0)

        bloklar = t_duz.view(-1, self.block_size)  # [num_blocks, block_size]

        # 1. Her blok için mutlak maksimum ölçek faktörü c1
        c1 = torch.max(torch.abs(bloklar), dim=-1, keepdim=True).values.clamp(min=1e-8)  # [num_blocks, 1]

        # 2. [-1, 1] aralığına normalize et
        bloklar_norm = bloklar / c1  # [num_blocks, block_size]

        # 3. En yakın NF4 seviyesine eşle (Quantization)
        # [num_blocks, block_size, 1] vs [1, 1, 16]
        farklar = torch.abs(bloklar_norm.unsqueeze(-1) - self.nf4_lut.view(1, 1, 16))
        q_indeksler = torch.argmin(farklar, dim=-1).to(torch.uint8)  # [num_blocks, block_size]

        if pad_miktari > 0:
            q_indeksler = q_indeksler.view(-1)[:N]

        return q_indeksler.view(orijinal_sekil), c1.squeeze(-1), orijinal_sekil

    def dekuantize_et(
        self,
        q_indeksler: torch.Tensor,
        c1: torch.Tensor,
        orijinal_sekil: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        4-bit NF4 indekslerini ve c1 ölçeklerini alarak orijinal FP tensörünü geri üretir.
        """
        N = q_indeksler.numel()
        pad_miktari = (self.block_size - (N % self.block_size)) % self.block_size

        q_duz = q_indeksler.flatten()
        if pad_miktari > 0:
            q_duz = torch.nn.functional.pad(q_duz, (0, pad_miktari), value=7)  # 7 -> 0.0

        bloklar_q = q_duz.view(-1, self.block_size)  # [num_blocks, block_size]
        lut = self.nf4_lut.to(device=q_indeksler.device)

        # İndekslerden FP değerlerine eşle
        bloklar_deq = lut[bloklar_q.long()]  # [num_blocks, block_size]

        # c1 ölçeği ile çarp
        c1_genisletilmis = c1.to(device=q_indeksler.device).unsqueeze(-1)  # [num_blocks, 1]
        dequant_bloklar = bloklar_deq * c1_genisletilmis

        dequant_duz = dequant_bloklar.view(-1)[:N]
        return dequant_duz.view(orijinal_sekil).to(dtype=dtype)


class DoubleQuantization:
    """
    Birincil ölçek faktörlerini (c1) ikincil 8-bit FP8/INT8 blok kuantizasyonu (B2=256)
    ile sıkıştıran motor. Bellek ek yükünü 0.5 bpp'den 0.127 bpp'ye indirir.
    """

    def __init__(self, block_size_2: int = 256):
        self.block_size_2 = block_size_2

    def c1_sikistir(self, c1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        c1 tensörünü 8-bit INT8 ve ikincil ölçek c2 ile sıkıştırır.
        Çıktı: (c1_int8, c2_skalalar, c1_offset)
        """
        N = c1.numel()
        pad = (self.block_size_2 - (N % self.block_size_2)) % self.block_size_2
        c1_pad = torch.nn.functional.pad(c1, (0, pad), value=0.0) if pad > 0 else c1

        bloklar = c1_pad.view(-1, self.block_size_2)
        c1_min = torch.min(bloklar, dim=-1, keepdim=True).values
        c1_max = torch.max(bloklar, dim=-1, keepdim=True).values
        c2 = (c1_max - c1_min).clamp(min=1e-8) / 255.0

        c1_int8 = torch.round((bloklar - c1_min) / c2).clamp(0, 255).to(torch.uint8)
        c1_int8 = c1_int8.view(-1)[:N]

        return c1_int8, c2.squeeze(-1), c1_min.squeeze(-1)

    def c1_coz(self, c1_int8: torch.Tensor, c2: torch.Tensor, c1_min: torch.Tensor) -> torch.Tensor:
        """Sıkıştırılmış c1_int8 tensörünü FP32 c1 ölçeklerine geri döndürür."""
        N = c1_int8.numel()
        pad = (self.block_size_2 - (N % self.block_size_2)) % self.block_size_2
        c1_pad = torch.nn.functional.pad(c1_int8, (0, pad), value=0) if pad > 0 else c1_int8

        bloklar_int8 = c1_pad.view(-1, self.block_size_2).float()
        c2_exp = c2.unsqueeze(-1)
        c1_min_exp = c1_min.unsqueeze(-1)

        c1_cozulmus = (bloklar_int8 * c2_exp) + c1_min_exp
        return c1_cozulmus.view(-1)[:N]
