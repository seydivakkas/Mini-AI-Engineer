"""
NT-Xent (Normalized Temperature-scaled Cross Entropy / InfoNCE) Kaybı
---------------------------------------------------------------------
SimCLR'ın temel kontrastif kayıp fonksiyonunun tam tensörel ve optimize edilmiş uygulaması.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    SimCLR için Normalized Temperature-scaled Cross Entropy (NT-Xent) Kaybı.
    
    Matematiksel Formül:
    --------------------
    N boyutlu batch için 2N adet artırılmış görünüm (z_i, z_j) üretilir.
    Pozitif bir çift (i, j) için kayıp:
    
    l_{i, j} = -log ( exp(sim(z_i, z_j) / tau) / sum_{k=1}^{2N} 1_{[k != i]} exp(sim(z_i, z_k) / tau) )
    
    Toplam Kayıp:
    L = 1 / (2N) sum_{k=1}^N [ l_{2k-1, 2k} + l_{2k, 2k-1} ]
    """
    def __init__(self, sicaklik: float = 0.5):
        super().__init__()
        if sicaklik <= 0.0:
            raise ValueError(f"Sıcaklık parametresi (tau) pozitif olmalıdır. Verilen: {sicaklik}")
        self.sicaklik = sicaklik

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Girdi:
        - z_i: 1. Görünüm projeksiyon tensörleri (N, D) - L2 normalize edilmiş
        - z_j: 2. Görünüm projeksiyon tensörleri (N, D) - L2 normalize edilmiş
        
        Çıktı:
        - skaler NT-Xent kaybı
        """
        N = z_i.size(0)
        cihaz = z_i.device
        
        # 1. Temsilleri birleştir: (2N, D)
        z = torch.cat([z_i, z_j], dim=0)
        
        # 2. Tüm çiftler arası kosinüs benzerlik matrisini hesapla: (2N, 2N)
        # z vektörleri zaten L2 normalize olduğu için z @ z.T kosinüs benzerliğidir.
        benzerlik_matrisi = torch.matmul(z, z.T) / self.sicaklik
        
        # 3. Kendisiyle olan benzerliği (köşegeni) maskele
        maske_kosegen = torch.eye(2 * N, dtype=torch.bool, device=cihaz)
        benzerlik_matrisi = benzerlik_matrisi.masked_fill(maske_kosegen, -1e9)
        
        # 4. Hedef etiketleri oluştur:
        # z_i (0 -> N-1) için pozitif eş z_j (N -> 2N-1)'dedir.
        # z_j (N -> 2N-1) için pozitif eş z_i (0 -> N-1)'dedir.
        etiketler = torch.cat([
            torch.arange(N, 2 * N, device=cihaz),
            torch.arange(0, N, device=cihaz)
        ], dim=0)
        
        # 5. Çapraz Entropi ile hesapla
        kayip = F.cross_entropy(benzerlik_matrisi, etiketler)
        return kayip

    def hesapla_hizalama_ve_duzenlilik(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> Dict[str, float]:
        """
        Wang & Isola (ICML 2020) Alignment ve Uniformity metriklerini hesaplar:
        - Alignment (Hizalama): Pozitif çiftlerin birbirine yakınlığı E[||z_i - z_j||^2]
        - Uniformity (Düzgünlük / İzotropi): Temsillerin küre üzerine homojen yayılımı log E[exp(-2 ||z_i - z_k||^2)]
        """
        with torch.no_grad():
            # 1. Alignment (Hizalama Hatası)
            hizalama = torch.mean(torch.norm(z_i - z_j, p=2, dim=1) ** 2).item()
            
            # 2. Pozitif ve Negatif Çift Ortalama Kosinüs Benzerlikleri
            pozitif_kosinus = torch.mean(torch.sum(z_i * z_j, dim=1)).item()
            
            # Negatif çiftler
            tum_z = torch.cat([z_i, z_j], dim=0)
            tum_sim = torch.matmul(tum_z, tum_z.T)
            N = z_i.size(0)
            
            # Negatif maskesi (köşegen ve pozitif çiftler hariç)
            negatif_maske = torch.ones_like(tum_sim, dtype=torch.bool)
            negatif_maske.fill_diagonal_(False)
            for k in range(N):
                negatif_maske[k, k + N] = False
                negatif_maske[k + N, k] = False
                
            negatif_kosinus = torch.mean(tum_sim[negatif_maske]).item()
            
            # Uniformity (Düzgün Yayılım)
            pdist_sq = torch.cdist(tum_z, tum_z, p=2) ** 2
            pdist_sq_maskeli = pdist_sq[negatif_maske]
            uniformity = torch.log(torch.mean(torch.exp(-2.0 * pdist_sq_maskeli)) + 1e-12).item()
            
        return {
            "alignment_loss": hizalama,
            "uniformity_loss": uniformity,
            "pozitif_kosinus_ort": pozitif_kosinus,
            "negatif_kosinus_ort": negatif_kosinus,
            "kosinus_marjini": pozitif_kosinus - negatif_kosinus
        }
