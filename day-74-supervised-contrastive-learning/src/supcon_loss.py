"""
Supervised Contrastive (SupCon) Kayıp Fonksiyonu
------------------------------------------------
Khosla et al. (NeurIPS 2020) makalesinde önerilen etiket destekli kontrastif
kayıp fonksiyonunun tam tensörel ve kararlı uygulaması.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    r"""
    Supervised Contrastive Loss (SupCon).
    
    Matematiksel Formülasyon:
    -------------------------
    L_{SupCon} = sum_{i in I} -1 / |P(i)| * sum_{p in P(i)} log [ exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau) ]
    
    - I: Minibatch'teki tüm çoklu görünümler kümesi (2N eleman).
    - A(i): i haricindeki tüm diğer örnekler kümesi (I \ {i}).
    - P(i): i ile AYNI sınıfa ait olan ve i'den farklı tüm pozitif örnekler kümesi.
    - |P(i)|: i için mevcut pozitif örnek sayısı.
    """
    def __init__(self, sicaklik: float = 0.1, baz_sicaklik: float = 0.07):
        super().__init__()
        if sicaklik <= 0.0:
            raise ValueError(f"Sıcaklık (tau) pozitif olmalıdır. Verilen: {sicaklik}")
        self.sicaklik = sicaklik
        self.baz_sicaklik = baz_sicaklik

    def forward(
        self,
        ozellikler: torch.Tensor,
        etiketler: torch.Tensor
    ) -> torch.Tensor:
        """
        Girdi:
        - ozellikler: (B, N_views, D) veya (2B, D) boyutunda L2 normalize edilmiş projeksiyon tensörleri
        - etiketler: (B,) boyutunda sınıf etiketleri
        
        Çıktı:
        - skaler SupCon kaybı
        """
        cihaz = ozellikler.device
        
        if ozellikler.dim() == 3:
            # (B, N_views, D) -> (B * N_views, D)
            batch_boyutu = ozellikler.size(0)
            gorunum_sayisi = ozellikler.size(1)
            z = torch.cat(torch.unbind(ozellikler, dim=1), dim=0)
            # Etiketleri görünümler kadar tekrarla: (B * N_views,)
            y = etiketler.repeat(gorunum_sayisi)
        elif ozellikler.dim() == 2:
            z = ozellikler
            y = etiketler
            gorunum_sayisi = 2
            batch_boyutu = ozellikler.size(0) // 2
        else:
            raise ValueError(f"Beklenen özellik boyutu 2D veya 3D, verilen: {ozellikler.shape}")

        toplam_ornek = z.size(0)
        
        # 1. Pozitif Eşleşme Maskesi Oluştur: M_ij = 1 iff (y_i == y_j and i != j)
        y_kolon = y.contiguous().view(-1, 1)
        ayni_sinif_maskesi = torch.eq(y_kolon, y_kolon.T).float().to(cihaz)
        
        # Kendisiyle olan eşleşmeyi (köşegeni) çıkar
        kosegen_maske = torch.scatter(
            torch.ones_like(ayni_sinif_maskesi),
            1,
            torch.arange(toplam_ornek, device=cihaz).view(-1, 1),
            0
        )
        pozitif_maskesi = ayni_sinif_maskesi * kosegen_maske

        # 2. Tüm çiftler arası kosinüs benzerlik matrisi: (toplam_ornek, toplam_ornek)
        dot_product = torch.matmul(z, z.T) / self.sicaklik

        # Sayısal kararlılık için her satırın maksimumunu çıkar
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        # 3. Payda (Denominator): exp_logits toplamı (köşegen hariç)
        exp_logits = torch.exp(logits) * kosegen_maske
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 4. Pozitif çiftler üzerinden log-olasılıkların ortalamasını al
        pozitif_sayilari = pozitif_maskesi.sum(1)
        # Sıfır pozitif içeren anchor örnekleri için maskele
        gecerli_anchor_maskesi = (pozitif_sayilari > 0)
        
        if not gecerli_anchor_maskesi.any():
            return torch.tensor(0.0, device=cihaz, requires_grad=True)

        mean_log_prob_pos = (pozitif_maskesi * log_prob).sum(1) / (pozitif_sayilari + 1e-12)

        # 5. Toplam Kayıp
        kayip = - (self.sicaklik / self.baz_sicaklik) * mean_log_prob_pos
        kayip = kayip[gecerli_anchor_maskesi].mean()
        return kayip

    def hesapla_geometrik_ayrisma(
        self,
        ozellikler: torch.Tensor,
        etiketler: torch.Tensor
    ) -> Dict[str, float]:
        """
        Sınıf içi vs Sınıflar arası ortalama kosinüs benzerliği ve ayrışma marjinini hesaplar.
        """
        with torch.no_grad():
            if ozellikler.dim() == 3:
                z = torch.cat(torch.unbind(ozellikler, dim=1), dim=0)
                y = etiketler.repeat(ozellikler.size(1))
            else:
                z = ozellikler
                y = etiketler
                
            N = z.size(0)
            sim_matrisi = torch.matmul(z, z.T)
            
            y_kolon = y.view(-1, 1)
            ayni_sinif = (y_kolon == y_kolon.T) & (~torch.eye(N, dtype=torch.bool, device=z.device))
            farkli_sinif = (y_kolon != y_kolon.T)
            
            sinif_ici = sim_matrisi[ayni_sinif].mean().item() if ayni_sinif.any() else 0.0
            siniflar_arasi = sim_matrisi[farkli_sinif].mean().item() if farkli_sinif.any() else 0.0
            
            return {
                "sinif_ici_kosinus": sinif_ici,
                "siniflar_arasi_kosinus": siniflar_arasi,
                "ayrisma_marjini": sinif_ici - siniflar_arasi
            }
