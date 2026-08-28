"""
Non-Parametrik k-NN Temsil Değerlendiricisi
-------------------------------------------
DINO ve SimCLR makalelerindeki gibi, sıfır parametre eğitimiyle (Zero-training)
temsil uzayının yerel manifold doğruluğunu ölçen sıcaklık ağırlıklı k-NN motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import torch
import torch.nn.functional as F


class KNNDegerlendirici:
    """
    Sıcaklık Ölçekli Ağırlıklı k-NN Sınıflandırıcısı (DINO / MoCo Protokolü).
    """
    def __init__(self, sicaklik: float = 0.07):
        self.sicaklik = sicaklik

    @torch.no_grad()
    def degerlendir(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        k_degerleri: List[int] = [1, 5, 10, 20],
        sinif_sayisi: int = 6
    ) -> Dict[str, float]:
        """
        Train gömülmelerini referans alarak Val sorgularının k-NN doğruluğunu hesaplar.
        
        Çıktı:
        - {"knn_top1_k1": float, "knn_top1_k5": float, ...}
        """
        # L2 Normalizasyon güvencesi
        x_train_norm = F.normalize(x_train, p=2, dim=1)
        x_val_norm = F.normalize(x_val, p=2, dim=1)

        N_val = x_val.size(0)
        max_k = max(k_degerleri)

        # Kosinüs benzerlik matrisi: (N_val, N_train)
        benzerlikler = torch.matmul(x_val_norm, x_train_norm.T)

        # En yakın max_k komşuyu bul
        topk_benzerlikler, topk_indeksler = torch.topk(benzerlikler, k=max_k, dim=1, largest=True, sorted=True)
        topk_etiketler = y_train[topk_indeksler] # (N_val, max_k)

        # Sıcaklık ağırlıkları: exp(sim / tau)
        topk_agirliklar = torch.exp(topk_benzerlikler / self.sicaklik) # (N_val, max_k)

        sonuclar = {}

        for k in k_degerleri:
            k_etiketler = topk_etiketler[:, :k]       # (N_val, k)
            k_agirliklar = topk_agirliklar[:, :k]     # (N_val, k)

            # Sınıf bazında ağırlıklı oy havuzu: (N_val, sinif_sayisi)
            oylar = torch.zeros(N_val, sinif_sayisi, device=x_val.device)
            for c in range(sinif_sayisi):
                maske = (k_etiketler == c).float()
                oylar[:, c] = (maske * k_agirliklar).sum(dim=1)

            tahminler = torch.argmax(oylar, dim=1)
            dogru = (tahminler == y_val).sum().item()
            acc = (dogru / max(1, N_val)) * 100.0
            sonuclar[f"knn_k_{k}"] = acc

        return sonuclar
