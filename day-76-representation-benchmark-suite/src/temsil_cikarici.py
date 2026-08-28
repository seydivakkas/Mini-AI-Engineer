"""
Dondurulmuş Temsil Çıkarıcı (Frozen Feature Extractor)
------------------------------------------------------
Önceden eğitilmiş modellerin omurgasını dondurarak (no_grad) veri kümesi üzerinden
yüksek boyutlu temsil vektörlerini (h) ve L2 normalize embedding'leri (e) çıkaran modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TemsilCikarici:
    """
    Dondurulmuş omurga ağından temsil vektörlerini bellek verimli çıkaran sınıf.
    """
    def __init__(self, model: nn.Module, cihaz: str = "cpu"):
        self.model = model.to(cihaz)
        self.cihaz = cihaz
        
        # Omurga parametrelerini dondur (Freeze)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def cikar(
        self,
        veri_yukleyici: DataLoader,
        normalize_et: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DataLoader üzerindeki tüm örneklerin temsil tensörünü ve etiketlerini döner.
        
        Çıktı:
        - temsiller: (N, D) boyutunda öznitelik tensörü
        - etiketler: (N,) boyutunda sınıf etiketleri
        """
        temsil_listesi = []
        etiket_listesi = []

        for batch in veri_yukleyici:
            # Girdi formatı: (x, y) veya (v1, v2, y)
            if len(batch) == 2:
                x, y = batch[0], batch[1]
            elif len(batch) >= 3:
                x, y = batch[0], batch[2]
            else:
                raise ValueError("Bilinmeyen batch formatı.")

            x = x.to(self.cihaz)
            
            # Model çıktısı (h, z) çifti dönebilir veya doğrudan h
            cikti = self.model(x)
            if isinstance(cikti, tuple):
                h = cikti[0]
            else:
                h = cikti

            # Eğer 4D tensörse (B, C, 1, 1) -> (B, C)
            if h.dim() > 2:
                h = h.view(h.size(0), -1)

            if normalize_et:
                h = F.normalize(h, p=2, dim=1)

            temsil_listesi.append(h.cpu())
            etiket_listesi.append(y.cpu())

        tum_temsiller = torch.cat(temsil_listesi, dim=0)
        tum_etiketler = torch.cat(etiket_listesi, dim=0)
        return tum_temsiller, tum_etiketler
