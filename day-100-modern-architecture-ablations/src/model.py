"""
Modern MiniViT Sınıflandırma Modeli (Day 100).
Ablasyon varyantlarını dinamik olarak inşa eden Hugging Face uyumlu Vision Transformer.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .konfigurasyon import ModernMiniViTConfig
from .modern_katmanlar import (
    RMSNorm,
    ModernTransformerBlok,
)


class YamaGomme(nn.Module):
    """Görüntüyü 2D yamalara bölüp D boyutlu vektörlere dönüştüren katman."""
    def __init__(self, config: ModernMiniViTConfig):
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
        x = self.projeksiyon(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ModernMiniViTForImageClassification(PreTrainedModel):
    """Modernize edilmiş Vision Transformer Sınıflandırma Modeli."""
    config_class = ModernMiniViTConfig
    base_model_prefix = "modern_minivit"
    main_input_name = "pixel_values"

    def __init__(self, config: ModernMiniViTConfig):
        super().__init__(config)
        self.config = config

        # 1. Yama Gömme
        self.yama_gomme = YamaGomme(config)
        self.yama_sayisi = self.yama_gomme.yama_sayisi

        # 2. CLS Token & Pozisyonel Kodlama
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.gizli_boyut))
        self.pozisyon_kodlama = nn.Parameter(torch.zeros(1, self.yama_sayisi + 1, config.gizli_boyut))
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # 3. Modern Transformer Blokları
        self.bloklar = nn.ModuleList([
            ModernTransformerBlok(config) for _ in range(config.katman_sayisi)
        ])

        # 4. Son Normalizasyon
        if config.norm_turu == "rmsnorm":
            self.final_norm = RMSNorm(config.gizli_boyut)
        else:
            self.final_norm = nn.LayerNorm(config.gizli_boyut)

        # 5. Sınıflandırma Başlığı
        self.siniflandirici = nn.Linear(config.gizli_boyut, config.sinif_sayisi)

        self.post_init()

    def _init_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if hasattr(module, "weight") and getattr(module.weight, "_is_hf_initialized", False):
            return

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
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

        # Yama gömme
        x = self.yama_gomme(pixel_values)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pozisyon_kodlama
        x = self.dropout(x)

        # Bloklardan geçiş
        for blok in self.bloklar:
            x = blok(x)

        # CLS temsili
        cls_temsili = x[:, 0]
        norm_temsili = self.final_norm(cls_temsili)
        logits = self.siniflandirici(norm_temsili)

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
