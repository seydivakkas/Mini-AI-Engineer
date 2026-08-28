"""
MiniViT-MoE v2 Konfigürasyon Modülü (Day 101 - Büyük Final).
Sparse Mixture of Experts (MoE) ve modern Vision Transformer parametreleri.
"""

from typing import Dict, Optional, Literal
from transformers import PretrainedConfig


class MiniViTMoEConfig(PretrainedConfig):
    """
    MiniViT Mixture of Experts (MoE) v2 için PretrainedConfig sınıfı.
    Top-K router, SwiGLU uzmanları ve aux load balancing loss parametrelerini barındırır.
    """
    model_type = "minivit_moe_v2"

    def __init__(
        self,
        goruntu_boyutu: int = 32,
        yama_boyutu: int = 4,
        kanal_sayisi: int = 3,
        gizli_boyut: int = 128,
        katman_sayisi: int = 4,
        dikkat_baslik_sayisi: int = 4,
        dropout: float = 0.0,
        sinif_sayisi: int = 10,
        initializer_range: float = 0.02,
        # MoE Parametreleri
        uzman_sayisi: int = 4,
        aktif_uzman_sayisi: int = 2,
        aux_loss_coef: float = 0.01,
        router_jitter_noise: float = 0.01,
        norm_turu: Literal["layernorm", "rmsnorm"] = "rmsnorm",
        dikkat_turu: Literal["standard", "sdpa"] = "sdpa",
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.goruntu_boyutu = goruntu_boyutu
        self.yama_boyutu = yama_boyutu
        self.kanal_sayisi = kanal_sayisi
        self.gizli_boyut = gizli_boyut
        self.katman_sayisi = katman_sayisi
        self.dikkat_baslik_sayisi = dikkat_baslik_sayisi
        self.dropout = dropout
        self.sinif_sayisi = sinif_sayisi
        self.initializer_range = initializer_range

        self.uzman_sayisi = uzman_sayisi
        self.aktif_uzman_sayisi = aktif_uzman_sayisi
        self.aux_loss_coef = aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.norm_turu = norm_turu
        self.dikkat_turu = dikkat_turu

        if id2label is None:
            cifar10_labels = [
                "uçak", "otomobil", "kuş", "kedi", "geyik",
                "köpek", "kurbağa", "at", "gemi", "kamyon"
            ]
            self.id2label = {i: label for i, label in enumerate(cifar10_labels)}
            self.label2id = {label: i for i, label in enumerate(cifar10_labels)}
        else:
            self.id2label = {int(k): v for k, v in id2label.items()}
            self.label2id = label2id or {v: int(k) for k, v in id2label.items()}
