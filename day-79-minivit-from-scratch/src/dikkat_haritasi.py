"""
Vision Transformer Dikkat Haritası ve Görsel Belirginlik Çıkarıcısı (Attention Rollout)
---------------------------------------------------------------------------------------
Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers" formülasyonuyla,
[CLS] token'ın görsel yamalarına olan dikkat akışını hesaplayıp 2D ısı haritasına dönüştüren modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class ViTDikkatCikarici:
    """
    MiniViT modelinden Attention Rollout ve 2D Uzamsal Belirginlik Haritası üreten sınıf.
    """
    @staticmethod
    def hesapla_attention_rollout(
        dikkat_listesi: List[torch.Tensor],
        kalinti_agirligi: float = 0.5
    ) -> torch.Tensor:
        """
        Tüm katmanlar boyunca dikkat akışını birleştirir:
        Rollout = Product_L ( 0.5 * MeanHead(A_l) + 0.5 * I )
        
        Girdi: L elemanlı [(Batch, Heads, Seq_Len, Seq_Len), ...]
        Çıktı: (Batch, Seq_Len, Seq_Len) kümülatif dikkat matrisi
        """
        # İlk katmanın başlar ortalaması: (B, N+1, N+1)
        b, _, n_tokens, _ = dikkat_listesi[0].shape
        I = torch.eye(n_tokens, device=dikkat_listesi[0].device).unsqueeze(0) # (1, N+1, N+1)

        kümülatif_rollout = torch.eye(n_tokens, device=dikkat_listesi[0].device).unsqueeze(0).repeat(b, 1, 1)

        for att in dikkat_listesi:
            # Başlar boyunca ortalama al: (B, N+1, N+1)
            att_mean = att.mean(dim=1)
            # Kalıntı bağlantıyı ekle (Identity shortcut)
            att_with_res = (1.0 - kalinti_agirligi) * att_mean + kalinti_agirligi * I
            # Normalizasyon
            att_norm = att_with_res / att_with_res.sum(dim=-1, keepdim=True)
            kümülatif_rollout = torch.matmul(att_norm, kümülatif_rollout)

        return kümülatif_rollout

    @classmethod
    def cls_dikkat_haritasi_2d(
        cls,
        dikkat_listesi: List[torch.Tensor],
        grid_boyutu: Tuple[int, int] = (8, 8),
        orijinal_boyut: Tuple[int, int] = (32, 32)
    ) -> np.ndarray:
        """
        [CLS] token'ın görsel yamalarına olan dikkatini 2D görüntü boyutuna interpole eder.
        
        Çıktı: (Batch, H, W) numpy ısı haritası
        """
        rollout = cls.hesapla_attention_rollout(dikkat_listesi) # (B, N+1, N+1)
        
        # [CLS] token'ın (0. indeks) diğer N yamaya olan dikkati: (B, N)
        cls_attn = rollout[:, 0, 1:] # (B, N)

        # 2D ızgaraya dönüştür: (B, 1, grid_h, grid_w)
        gh, gw = grid_boyutu
        cls_2d = cls_attn.view(-1, 1, gh, gw)

        # Orijinal görsel çözünürlüğüne (H, W) bilinear büyüt
        cls_upsampled = F.interpolate(
            cls_2d,
            size=orijinal_boyut,
            mode="bilinear",
            align_corners=False
        )

        # Normalize et: [0, 1] aralığı
        min_v = cls_upsampled.amin(dim=(2, 3), keepdim=True)
        max_v = cls_upsampled.amax(dim=(2, 3), keepdim=True)
        cls_norm = (cls_upsampled - min_v) / (max_v - min_v + 1e-12)

        return cls_norm.squeeze(1).detach().cpu().numpy()
