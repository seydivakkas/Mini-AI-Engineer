"""
Sıfırdan Transformer Encoder Bloğu ve Çok Katmanlı Encoder Gövdesi
------------------------------------------------------------------
Pre-LayerNorm ve Post-LayerNorm mimarilerini destekleyen, Residual bağlantılı,
MHSA ve FFN içeren üretim seviyesinde Transformer Encoder modülü.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, List, Optional, Union
import torch
import torch.nn as nn

from .multi_head_attention import CokKafaliOzDikkat
from .layer_norm import OzelLayerNorm
from .feed_forward import BeslemeliIleriAg
from .pozisyonel_kodlama import SinusoidalPozisyonelKodlama, OgrenilebilirPozisyonelKodlama


class TransformerEncoderBlogu(nn.Module):
    """
    Tek bir Transformer Encoder Bloğu:
    Pre-LN:  x = x + MHSA(LN1(x)),  x = x + FFN(LN2(x))
    Post-LN: x = LN1(x + MHSA(x)),  x = LN2(x + FFN(x))
    """
    def __init__(
        self,
        model_boyutu: int = 64,
        kafa_sayisi: int = 4,
        genisleme_faktoru: int = 4,
        dropout_orani: float = 0.1,
        norm_tipi: str = "pre_ln",
        aktivasyon: str = "gelu"
    ):
        super().__init__()
        self.norm_tipi = norm_tipi.lower()
        assert self.norm_tipi in ["pre_ln", "post_ln"], "norm_tipi 'pre_ln' veya 'post_ln' olmalıdır."

        self.dikkat = CokKafaliOzDikkat(
            model_boyutu=model_boyutu,
            kafa_sayisi=kafa_sayisi,
            dropout_orani=dropout_orani
        )
        self.ln1 = OzelLayerNorm(model_boyutu)
        self.ln2 = OzelLayerNorm(model_boyutu)
        
        self.ffn = BeslemeliIleriAg(
            model_boyutu=model_boyutu,
            genisleme_faktoru=genisleme_faktoru,
            dropout_orani=dropout_orani,
            aktivasyon=aktivasyon
        )
        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Girdi: (Batch, Seq_Len, Model_Boyutu)
        Çıktı: (Batch, Seq_Len, Model_Boyutu), (Batch, Heads, Seq_Len, Seq_Len)
        """
        if self.norm_tipi == "pre_ln":
            # 1. Pre-LN Dikkat Bloğu (Modern Standart - GPT/ViT)
            norm_x = self.ln1(x)
            attn_out, dikkat_haritalari = self.dikkat(norm_x, mask=mask)
            x = x + self.dropout(attn_out)

            # 2. Pre-LN FFN Bloğu
            norm_x2 = self.ln2(x)
            ffn_out = self.ffn(norm_x2)
            x = x + self.dropout(ffn_out)

        else:
            # 1. Post-LN Dikkat Bloğu (Orijinal Vaswani et al. 2017)
            attn_out, dikkat_haritalari = self.dikkat(x, mask=mask)
            x = self.ln1(x + self.dropout(attn_out))

            # 2. Post-LN FFN Bloğu
            ffn_out = self.ffn(x)
            x = self.ln2(x + self.dropout(ffn_out))

        return x, dikkat_haritalari


class TransformerEncoderGovdesi(nn.Module):
    """
    Çok Katmanlı Transformer Encoder Yığını (L Katmanlı Omurga).
    """
    def __init__(
        self,
        katman_sayisi: int = 4,
        model_boyutu: int = 64,
        kafa_sayisi: int = 4,
        genisleme_faktoru: int = 4,
        maksimum_uzunluk: int = 512,
        dropout_orani: float = 0.1,
        norm_tipi: str = "pre_ln",
        pozisyon_tipi: str = "sinusoidal",
        aktivasyon: str = "gelu"
    ):
        super().__init__()
        self.norm_tipi = norm_tipi.lower()
        
        # Pozisyonel Kodlama
        if pozisyon_tipi.lower() == "sinusoidal":
            self.pos_encoder = SinusoidalPozisyonelKodlama(model_boyutu, maksimum_uzunluk, dropout_orani)
        else:
            self.pos_encoder = OgrenilebilirPozisyonelKodlama(model_boyutu, maksimum_uzunluk, dropout_orani)

        # L adet Encoder Bloğu Yığını
        self.bloklar = nn.ModuleList([
            TransformerEncoderBlogu(
                model_boyutu=model_boyutu,
                kafa_sayisi=kafa_sayisi,
                genisleme_faktoru=genisleme_faktoru,
                dropout_orani=dropout_orani,
                norm_tipi=norm_tipi,
                aktivasyon=aktivasyon
            )
            for _ in range(katman_sayisi)
        ])

        # Pre-LN için en son nihai LayerNorm katmanı
        self.son_ln = OzelLayerNorm(model_boyutu) if self.norm_tipi == "pre_ln" else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        tum_katmanlari_don: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Çıktılar:
        - cikti: (Batch, Seq_Len, Model_Boyutu)
        - dikkat_haritalari: L elemanlı liste [(B, H, N, N), ...]
        - ara_katman_ciktilari: (Opsiyonel) L elemanlı liste
        """
        x = self.pos_encoder(x)
        dikkat_listesi = []
        katman_ciktilari = []

        for blok in self.bloklar:
            x, att = blok(x, mask=mask)
            dikkat_listesi.append(att)
            if tum_katmanlari_don:
                katman_ciktilari.append(x)

        x = self.son_ln(x)
        return x, dikkat_listesi, katman_ciktilari if tum_katmanlari_don else None
