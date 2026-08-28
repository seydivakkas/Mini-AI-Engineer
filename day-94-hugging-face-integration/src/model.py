"""
Day 94: Hugging Face PreTrainedModel Tabanlı MiniViT Modeli
----------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .konfigurasyon import MiniViTConfig


class YamaGomme(nn.Module):
    """Görüntüyü yamalara (patches) bölüp doğrusal projeksiyonla gömer."""

    def __init__(self, goruntu_boyutu: int, yama_boyutu: int, giris_kanali: int, gizli_boyut: int):
        super().__init__()
        self.goruntu_boyutu = goruntu_boyutu
        self.yama_boyutu = yama_boyutu
        self.yama_sayisi = (goruntu_boyutu // yama_boyutu) ** 2

        self.projeksiyon = nn.Conv2d(
            in_channels=giris_kanali,
            out_channels=gizli_boyut,
            kernel_size=yama_boyutu,
            stride=yama_boyutu,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        x = self.projeksiyon(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MiniViTForImageClassification(PreTrainedModel):
    """
    Hugging Face `PreTrainedModel` sınıfından türetilmiş,
    SafeTensors ve AutoModel uyumlu Vision Transformer görsel sınıflandırıcısı.
    """

    config_class = MiniViTConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MiniViTConfig):
        super().__init__(config)
        self.config = config

        # 1. Yama Gömme
        self.yama_gomme = YamaGomme(
            goruntu_boyutu=config.goruntu_boyutu,
            yama_boyutu=config.yama_boyutu,
            giris_kanali=config.giris_kanali,
            gizli_boyut=config.gizli_boyut,
        )
        toplam_yama = self.yama_gomme.yama_sayisi

        # 2. CLS Token ve Pozisyonel Gömme
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.gizli_boyut))
        self.pos_embedding = nn.Parameter(torch.zeros(1, toplam_yama + 1, config.gizli_boyut))
        self.pos_dropout = nn.Dropout(config.dropout_orani)

        # 3. Transformer Encoder Blokları (Pre-LayerNorm)
        encoder_katmani = nn.TransformerEncoderLayer(
            d_model=config.gizli_boyut,
            nhead=config.dikkat_baslik_sayisi,
            dim_feedforward=config.mlp_ara_boyut,
            dropout=config.dropout_orani,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_katmani,
            num_layers=config.katman_sayisi,
        )

        # 4. Normalizasyon ve Sınıflandırma Başlığı
        self.norm = nn.LayerNorm(config.gizli_boyut)
        self.siniflandirici = nn.Linear(config.gizli_boyut, config.sinif_sayisi)

        # Ağırlık başlatma
        self.post_init()

    def _init_weights(self, module: nn.Module):
        """Hugging Face standart ağırlık ilklendirmesi."""
        if getattr(module, "_is_hf_initialized", False):
            return
        if hasattr(module, "weight") and getattr(module.weight, "_is_hf_initialized", False):
            return

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], ImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

        if pixel_values is None:
            raise ValueError("`pixel_values` tensörü boş olamaz.")

        batch_size = pixel_values.shape[0]

        # 1. Yamalara Böl ve CLS Token Ekle
        x = self.yama_gomme(pixel_values)  # [B, N, D]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        # 2. Pozisyonel Gömme Ekle
        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        # 3. Transformer Encoder
        x = self.encoder(x)  # [B, N+1, D]

        # 4. CLS Token Çıkışını Havuzla ve Sınıflandır
        cls_cikti = self.norm(x[:, 0])  # [B, D]
        logits = self.siniflandirici(cls_cikti)  # [B, NumLabels]

        # 5. Kayıp Hesabı (Eğer Etiket Verilmişse)
        loss = None
        if labels is not None:
            kayip_fonk = nn.CrossEntropyLoss()
            loss = kayip_fonk(logits.view(-1, self.config.sinif_sayisi), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
