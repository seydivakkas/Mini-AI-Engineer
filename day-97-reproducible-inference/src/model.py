"""
MiniViT Model Mimarisi Modülü (Day 97 - Reproducible Inference).
Hugging Face PreTrainedModel tabanlı Vision Transformer sınıflandırıcı.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .konfigurasyon import MiniViTConfig


class YamaGomme(nn.Module):
    """Görüntüyü 2D yamalara bölüp D boyutlu vektörlere dönüştüren katman."""
    def __init__(self, config: MiniViTConfig):
        super().__init__()
        self.goruntu_boyutu = config.goruntu_boyutu
        self.yama_boyutu = config.yama_boyutu
        self.kanal_sayisi = config.kanal_sayisi
        self.gizli_boyut = config.gizli_boyut

        assert self.goruntu_boyutu % self.yama_boyutu == 0, "Görüntü boyutu yama boyutuna tam bölünmelidir."
        self.yama_sayisi = (self.goruntu_boyutu // self.yama_boyutu) ** 2

        self.projeksiyon = nn.Conv2d(
            in_channels=self.kanal_sayisi,
            out_channels=self.gizli_boyut,
            kernel_size=self.yama_boyutu,
            stride=self.yama_boyutu,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        x = self.projeksiyon(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MiniViTForImageClassification(PreTrainedModel):
    """Hugging Face standartlarına uyumlu Mini Vision Transformer Sınıflandırma Modeli."""
    config_class = MiniViTConfig
    base_model_prefix = "minivit"
    main_input_name = "pixel_values"

    def __init__(self, config: MiniViTConfig):
        super().__init__(config)
        self.config = config

        # 1. Yama Gömme
        self.yama_gomme = YamaGomme(config)
        self.yama_sayisi = self.yama_gomme.yama_sayisi

        # 2. CLS Token ve Pozisyonel Kodlama
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.gizli_boyut))
        self.pozisyon_kodlama = nn.Parameter(torch.zeros(1, self.yama_sayisi + 1, config.gizli_boyut))
        self.dropout = nn.Dropout(config.dropout)

        # 3. Transformer Encoder Bloğu
        katman = nn.TransformerEncoderLayer(
            d_model=config.gizli_boyut,
            nhead=config.dikkat_baslik_sayisi,
            dim_feedforward=config.ileri_besleme_boyutu,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(katman, num_layers=config.katman_sayisi)

        # 4. Sınıflandırma Başlığı
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

        # 2. Pozisyonel Kodlama Ekle
        x = x + self.pozisyon_kodlama
        x = self.dropout(x)

        # 3. Transformer Encoder
        x = self.encoder(x)

        # 4. Sınıflandırma
        cls_temsili = x[:, 0]  # CLS token [B, D]
        norm_temsili = self.norm(cls_temsili)
        logits = self.siniflandirici(norm_temsili)  # [B, num_classes]

        loss = None
        if labels is not None:
            kayip_fonk = nn.CrossEntropyLoss()
            loss = kayip_fonk(logits.view(-1, self.config.sinif_sayisi), labels.view(-1))

        if not return_dict:
            cikis = (logits,)
            return ((loss,) + cikis) if loss is not None else cikis

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
