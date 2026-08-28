"""
MiniViT-MoE v2 Sınıflandırma Modeli (Day 101).
Hugging Face PreTrainedModel tabanlı Sparse Mixture of Experts Vision Transformer.
"""

from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .konfigurasyon import MiniViTMoEConfig
from .moe_katmanlari import (
    RMSNorm,
    MoETransformerBlok,
)


class YamaGomme(nn.Module):
    """Görüntüyü 2D yamalara bölüp D boyutlu vektörlere dönüştüren katman."""
    def __init__(self, config: MiniViTMoEConfig):
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


class MiniViTMoEForImageClassification(PreTrainedModel):
    """
    MiniViT-MoE v2 Hugging Face Uyumlu Görüntü Sınıflandırma Modeli.
    Sparse MoE katmanları ile parametre kapasitesini 4x artırırken FLOPs maliyetini minimumda tutar.
    """
    config_class = MiniViTMoEConfig
    base_model_prefix = "minivit_moe_v2"
    main_input_name = "pixel_values"

    def __init__(self, config: MiniViTMoEConfig):
        super().__init__(config)
        self.config = config

        # 1. Yama Gömme
        self.yama_gomme = YamaGomme(config)
        self.yama_sayisi = self.yama_gomme.yama_sayisi

        # 2. CLS Token & Pozisyonel Kodlama
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.gizli_boyut))
        self.pozisyon_kodlama = nn.Parameter(torch.zeros(1, self.yama_sayisi + 1, config.gizli_boyut))
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # 3. MoE Transformer Blokları
        self.bloklar = nn.ModuleList([
            MoETransformerBlok(config) for _ in range(config.katman_sayisi)
        ])

        # 4. Final Normalizasyon
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

    def aktif_parametre_hesapla(self) -> Dict[str, int]:
        """Modelin toplam ve her çıkarım adımında aktif olan parametre sayısını döndürür."""
        toplam_param = sum(p.numel() for p in self.parameters())

        # Tek bir MoE katmanındaki uzman parametreleri
        uzman_param_toplami = 0
        for blok in self.bloklar:
            for uzman in blok.moe_katmani.uzmanlar:
                uzman_param_toplami += sum(p.numel() for p in uzman.parameters())

        tek_uzman_param = uzman_param_toplami // (self.config.katman_sayisi * self.config.uzman_sayisi)
        pasif_uzman_param = (self.config.uzman_sayisi - self.config.aktif_uzman_sayisi) * tek_uzman_param * self.config.katman_sayisi
        aktif_param = toplam_param - pasif_uzman_param

        return {
            "toplam_parametre": toplam_param,
            "aktif_parametre": aktif_param,
            "tasarruf_orani_yuzde": round(((toplam_param - aktif_param) / toplam_param) * 100, 2),
        }

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

        # MoE Bloklarından geçiş ve Aux Loss toplama
        toplam_aux_loss = torch.tensor(0.0, device=pixel_values.device)
        for blok in self.bloklar:
            x, aux_loss = blok(x)
            toplam_aux_loss = toplam_aux_loss + aux_loss

        # CLS temsili
        cls_temsili = x[:, 0]
        norm_temsili = self.final_norm(cls_temsili)
        logits = self.siniflandirici(norm_temsili)

        loss = None
        if labels is not None:
            kayip_fonk = nn.CrossEntropyLoss()
            gorev_kaybi = kayip_fonk(logits.view(-1, self.config.sinif_sayisi), labels.view(-1))
            loss = gorev_kaybi + (self.config.aux_loss_coef * toplam_aux_loss)

        if not return_dict:
            cikis = (logits,)
            return ((loss,) + cikis) if loss is not None else cikis

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
