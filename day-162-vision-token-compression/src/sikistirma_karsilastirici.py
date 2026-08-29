"""
Görüntü Token Sıkıştırma Yöntemleri Karşılaştırıcı Modülü (Day 162 - FAZ 9).
3 Sıkıştırma yönteminin token sayısı, bellek kazancı ve hızını kıyaslar.
"""

from typing import Dict, Any
import torch
import time

from .qformer_sikistirici import QFormerSikistirici
from .c_abstractor_sikistirici import CAbstractorSikistirici
from .spatial_pooling_sikistirici import SpatialPoolingSikistirici


class SikistirmaKarsilastirici:
    """3 Token Sıkıştırma Yönteminin Analitik Kıyaslayıcısı."""

    @classmethod
    def yontemleri_karsilastir(cls, batch_size: int = 4) -> Dict[str, Any]:
        """Tüm yöntemleri 256 ham token üzerinde test eder."""
        dummy_visual = torch.randn(batch_size, 256, 768)

        modeller = {
            "1. Ham ViT (Sıkıştırmasız)": None,
            "2. Spatial Pooling (2x2)": SpatialPoolingSikistirici(pool_boyutu=2),
            "3. C-Abstractor (Conv 2x)": CAbstractorSikistirici(stride=2),
            "4. BLIP-2 Q-Former (32 Query)": QFormerSikistirici(num_query_tokens=32),
        }

        sonuclar = {}

        for isim, model in modeller.items():
            if model is None:
                token_sayisi = 256
                sikistirma_orani = 0.0
                bellek_tasarrufu = 0.0
                sure_ms = 0.05
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    out = model(dummy_visual)
                end = time.perf_counter()
                token_sayisi = out.shape[1]
                sikistirma_orani = (1.0 - token_sayisi / 256.0) * 100.0
                bellek_tasarrufu = (1.0 - (token_sayisi / 256.0) ** 2) * 100.0  # Self-attention O(N^2)
                sure_ms = (end - start) * 1000.0

            sonuclar[isim] = {
                "token_sayisi": token_sayisi,
                "sikistirma_orani": round(sikistirma_orani, 1),
                "attention_bellek_tasarrufu": round(bellek_tasarrufu, 1),
                "islem_suresi_ms": round(sure_ms, 3),
            }

        return sonuclar
