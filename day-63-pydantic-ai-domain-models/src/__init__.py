"""
Day 63: Pydantic v2 ile Tip Güvenli Girdi/Çıktı Sözleşmeleri ve AI Domain Modelleri Paketi.
"""

from .domain_modelleri import (
    GorselMetadatasi,
    BoundingBoxModeli,
    NesneTespitiSonucu,
    VektorEmbeddingSozlesmesi,
    InferenceIstekSozlesmesi,
    InferenceYanitSozlesmesi
)
from .sozlesme_dogrulayici import SozlesmeDogrulayici, PydanticBenchmarkEngine
from .gorsellestirici import PydanticGorsellestirici

__all__ = [
    "GorselMetadatasi",
    "BoundingBoxModeli",
    "NesneTespitiSonucu",
    "VektorEmbeddingSozlesmesi",
    "InferenceIstekSozlesmesi",
    "InferenceYanitSozlesmesi",
    "SozlesmeDogrulayici",
    "PydanticBenchmarkEngine",
    "PydanticGorsellestirici"
]
