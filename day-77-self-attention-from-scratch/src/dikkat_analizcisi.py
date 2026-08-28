"""
Dikkat Mekanizması Analizcisi (Attention Inspector & Analyzer)
--------------------------------------------------------------
Dikkat haritalarının entropisini, başlar arası çeşitliliği (Head Diversity),
ortalama dikkat mesafesini ve sqrt(d_k) ölçekleme etkisini analiz eden araç.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F


class DikkatAnalizcisi:
    """
    Çok kafalı dikkat ağırlık matrislerini niceliksel ve geometrik olarak analiz eden sınıf.
    """
    @staticmethod
    def hesapla_dikkat_entropisi(dikkat_haritalari: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
        """
        Her bir dikkat başı için ortalama Shannon entropisini hesaplar.
        
        Girdi: (Batch, Heads, Seq_Len, Seq_Len)
        Çıktı: (Heads,) boyutunda ortalama entropi dizisi
        """
        # A: (B, H, N, N)
        A = dikkat_haritalari.detach().cpu()
        entropi = - (A * torch.log(A + eps)).sum(dim=-1) # (B, H, N)
        ortalama_entropi = entropi.mean(dim=(0, 2)).numpy() # (H,)
        return ortalama_entropi

    @staticmethod
    def hesapla_dikkat_mesafesi(dikkat_haritalari: torch.Tensor) -> np.ndarray:
        """
        Her başın ortalama dikkat mesafesini (|i - j|) hesaplar (Lokal vs Global Başlar).
        
        Girdi: (Batch, Heads, Seq_Len, Seq_Len)
        Çıktı: (Heads,) boyutunda ortalama piksel/token mesafesi
        """
        A = dikkat_haritalari.detach().cpu()
        _, H, N, _ = A.shape
        i_ind = torch.arange(N).view(1, 1, N, 1)
        j_ind = torch.arange(N).view(1, 1, 1, N)
        mesafe_matrisi = torch.abs(i_ind - j_ind).float() # (1, 1, N, N)

        ortalama_mesafe = (A * mesafe_matrisi).sum(dim=-1).mean(dim=(0, 2)).numpy() # (H,)
        return ortalama_mesafe

    @staticmethod
    def hesapla_baslar_arasi_cesitlilik(dikkat_haritalari: torch.Tensor) -> float:
        """
        Farklı başların dikkat matrisleri arasındaki ortalama Kosinüs Mesafesini (1 - CosSim) hesaplar.
        Yüksek değer = Başlar birbirinden bağımsız ve çeşitli desenler öğreniyor.
        """
        A = dikkat_haritalari.detach().cpu().mean(dim=0) # (H, N, N)
        H, N, _ = A.shape
        A_flat = A.view(H, N * N)
        A_norm = F.normalize(A_flat, p=2, dim=1)

        sim_mat = torch.matmul(A_norm, A_norm.T) # (H, H)
        # Köşegen dışı elemanların ortalama benzerliği
        maske = ~torch.eye(H, dtype=torch.bool)
        if maske.sum() > 0:
            ort_sim = sim_mat[maske].mean().item()
            cesitlilik = 1.0 - ort_sim
        else:
            cesitlilik = 0.0
        return float(cesitlilik)

    @staticmethod
    def olcek_etkisi_analizi(d_k: int = 64, seq_len: int = 16) -> Dict[str, Any]:
        """
        sqrt(d_k) ile ölçeklemenin Softmax doygunluğuna (Saturation) ve gradyana etkisini modeller.
        """
        torch.manual_seed(42)
        q = torch.randn(1, 1, seq_len, d_k)
        k = torch.randn(1, 1, seq_len, d_k)

        # 1. Ölçeksiz (Unscaled)
        skor_olceksiz = torch.matmul(q, k.transpose(-1, -2))
        softmax_olceksiz = F.softmax(skor_olceksiz, dim=-1)
        entropi_olceksiz = - (softmax_olceksiz * torch.log(softmax_olceksiz + 1e-12)).sum(dim=-1).mean().item()

        # 2. Ölçekli (Scaled by sqrt(d_k))
        skor_olcekli = skor_olceksiz / np.sqrt(d_k)
        softmax_olcekli = F.softmax(skor_olcekli, dim=-1)
        entropi_olcekli = - (softmax_olcekli * torch.log(softmax_olcekli + 1e-12)).sum(dim=-1).mean().item()

        return {
            "olceksiz_skor_std": float(skor_olceksiz.std().item()),
            "olcekli_skor_std": float(skor_olcekli.std().item()),
            "olceksiz_entropi": entropi_olceksiz,
            "olcekli_entropi": entropi_olcekli,
            "skor_olceksiz_ornek": skor_olceksiz[0, 0, 0].numpy(),
            "skor_olcekli_ornek": skor_olcekli[0, 0, 0].numpy(),
        }
