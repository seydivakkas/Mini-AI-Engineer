"""
MiniViT Konfigürasyon Modülü (Day 96 - Public Release v1.0).
Hugging Face PretrainedConfig tabanlı üretim konfigürasyon sınıfı.
"""

from typing import Dict, Optional
from transformers import PretrainedConfig


class MiniViTConfig(PretrainedConfig):
    """
    Mini Vision Transformer v1.0 için PretrainedConfig sınıfı.
    Model mimarisinin tüm hiperparametrelerini ve sınıf etiketlerini saklar.
    """
    model_type = "minivit"

    def __init__(
        self,
        goruntu_boyutu: int = 32,
        yama_boyutu: int = 4,
        kanal_sayisi: int = 3,
        gizli_boyut: int = 128,
        katman_sayisi: int = 4,
        dikkat_baslik_sayisi: int = 4,
        ileri_besleme_boyutu: int = 256,
        dropout: float = 0.1,
        sinif_sayisi: int = 10,
        initializer_range: float = 0.02,
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
        self.ileri_besleme_boyutu = ileri_besleme_boyutu
        self.dropout = dropout
        self.sinif_sayisi = sinif_sayisi
        self.initializer_range = initializer_range

        # Varsayılan CIFAR-10 etiket haritaları
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
