"""
Day 94: Hugging Face Uyumlu MiniViT Konfigürasyon Sınıfı (PretrainedConfig)
--------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Optional
from transformers import PretrainedConfig


class MiniViTConfig(PretrainedConfig):
    """
    MiniViT (Vision Transformer) modeli için Hugging Face `PretrainedConfig` sınıfı.
    `model_type = "minivit"` olarak kayıt edilir ve `AutoConfig` ile tam uyumludur.
    """

    model_type = "minivit"

    def __init__(
        self,
        goruntu_boyutu: int = 32,
        yama_boyutu: int = 4,
        giris_kanali: int = 3,
        gizli_boyut: int = 128,
        katman_sayisi: int = 4,
        dikkat_baslik_sayisi: int = 4,
        mlp_ara_boyut: int = 256,
        dropout_orani: float = 0.1,
        sinif_sayisi: int = 10,
        initializer_range: float = 0.02,
        id2label: Optional[Dict[str, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )
        self.goruntu_boyutu = goruntu_boyutu
        self.yama_boyutu = yama_boyutu
        self.giris_kanali = giris_kanali
        self.gizli_boyut = gizli_boyut
        self.katman_sayisi = katman_sayisi
        self.dikkat_baslik_sayisi = dikkat_baslik_sayisi
        self.mlp_ara_boyut = mlp_ara_boyut
        self.dropout_orani = dropout_orani
        self.sinif_sayisi = sinif_sayisi
        self.initializer_range = initializer_range

        # Varsayılan sınıf etiketleri
        if self.id2label is None:
            self.id2label = {str(i): f"SINIF_{i}" for i in range(sinif_sayisi)}
            self.label2id = {f"SINIF_{i}": i for i in range(sinif_sayisi)}
        elif self.label2id is None:
            self.label2id = {v: int(k) for k, v in self.id2label.items()}
