"""
Token Paketleme (Token Packing / Multipack) ve Blok-Diyagonal Maske Modülü (Day 106).
First-Fit Decreasing (FFD) Bin-Packing, Prompt Maskeleme (-100) ve Pozisyon Sıfırlama.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch


@dataclass
class Ornek:
    """Tek bir talimat-cevap (Instruction-Response) örneği."""
    prompt_ids: List[int]
    response_ids: List[int]

    @property
    def toplam_uzunluk(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


@dataclass
class PaketlenmisDizi:
    """Tek bir max_seq_len içine paketlenmiş birden fazla örnek."""
    input_ids: torch.Tensor       # [max_seq_len]
    labels: torch.Tensor          # [max_seq_len] (Prompt'lar -100 ile maskeli)
    position_ids: torch.Tensor    # [max_seq_len] (Her örnek başında 0'a sıfırlanır)
    ornek_uzunluklari: List[int]  # [N_1, N_2, ...]
    cu_seqlens: torch.Tensor      # [0, N_1, N_1 + N_2, ...]
    doluluk_orani: float          # Efektif token / max_seq_len


def olustur_blok_diyagonal_maske(
    ornek_uzunluklari: List[int],
    toplam_uzunluk: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Paketlenmiş dizide örnekler arası dikkat sızıntısını (Cross-Contamination) önleyen
    blok-diyagonal nedensel dikkat maskesi (Block-Diagonal Causal Mask) üretir.
    """
    maske = torch.full((toplam_uzunluk, toplam_uzunluk), float("-inf"), device=device)

    baslangic = 0
    for uzunluk in ornek_uzunluklari:
        bitis = baslangic + uzunluk
        # Alt-örnek içinde standart nedensel üçgen maske
        alt_maske = torch.triu(torch.full((uzunluk, uzunluk), float("-inf"), device=device), diagonal=1)
        maske[baslangic:bitis, baslangic:bitis] = alt_maske
        baslangic = bitis

    return maske


class TokenPaketleyici:
    """
    First-Fit Decreasing (FFD) Bin-Packing algoritmasıyla örnekleri paketleyen motor.
    Sıfır padding kaybı ve maksimum GPU verimi sağlar.
    """

    def __init__(self, max_seq_len: int = 2048, pad_token_id: int = 0):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def paketle(self, ornekler: List[Ornek]) -> List[PaketlenmisDizi]:
        """
        Örnekleri FFD algoritmasıyla sıralayıp torbalara (bins) paketler.
        """
        # 1. Uzunluğa göre azalan sıralama (First-Fit Decreasing)
        sirali_ornekler = sorted(ornekler, key=lambda x: x.toplam_uzunluk, reverse=True)

        torbalar: List[List[Ornek]] = []
        torba_kalan_kapasite: List[int] = []

        for o in sirali_ornekler:
            if o.toplam_uzunluk > self.max_seq_len:
                # max_seq_len'den büyükse kırp
                kirpilmis = Ornek(
                    prompt_ids=o.prompt_ids[:self.max_seq_len // 2],
                    response_ids=o.response_ids[:self.max_seq_len - len(o.prompt_ids[:self.max_seq_len // 2])],
                )
                o = kirpilmis

            yerlesti = False
            for idx, kalan in enumerate(torba_kalan_kapasite):
                if o.toplam_uzunluk <= kalan:
                    torbalar[idx].append(o)
                    torba_kalan_kapasite[idx] -= o.toplam_uzunluk
                    yerlesti = True
                    break

            if not yerlesti:
                torbalar.append([o])
                torba_kalan_kapasite.append(self.max_seq_len - o.toplam_uzunluk)

        # 2. Torbaları PaketlenmisDizi nesnelerine dönüştür
        paketlenmis_diziler: List[PaketlenmisDizi] = []

        for torba in torbalar:
            inp_list = []
            lbl_list = []
            pos_list = []
            lens = []

            for o in torba:
                n = o.toplam_uzunluk
                lens.append(n)
                inp_list.extend(o.prompt_ids + o.response_ids)
                # Prompt token'ları -100 ile maskelenir (SFT Loss Masking)
                lbl_list.extend([-100] * len(o.prompt_ids) + o.response_ids)
                # Pozisyon ID'leri her alt-örnek için 0'dan başlar
                pos_list.extend(list(range(n)))

            efektif_len = len(inp_list)
            pad_len = self.max_seq_len - efektif_len

            if pad_len > 0:
                inp_list.extend([self.pad_token_id] * pad_len)
                lbl_list.extend([-100] * pad_len)
                pos_list.extend([0] * pad_len)

            cu_seqlens = [0]
            curr = 0
            for l in lens:
                curr += l
                cu_seqlens.append(curr)

            dizi = PaketlenmisDizi(
                input_ids=torch.tensor(inp_list, dtype=torch.long),
                labels=torch.tensor(lbl_list, dtype=torch.long),
                position_ids=torch.tensor(pos_list, dtype=torch.long),
                ornek_uzunluklari=lens,
                cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
                doluluk_orani=float(efektif_len / self.max_seq_len),
            )
            paketlenmis_diziler.append(dizi)

        return paketlenmis_diziler
