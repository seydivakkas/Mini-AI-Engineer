"""
Day 70: Mixup, CutMix ve Label Smoothing Modern Düzenlileştirme Paketi
"""

from src.mixup_cutmix import ModernArtirici
from src.kayip_fonksiyonlari import YumusatilmisCrossEntropyLoss
from src.deney_modeli import ModernRegulerVisionNet
from src.reguler_karsilastirici import RegulerizasyonLaboratuvari
from src.gorsellestirici import RegulerizasyonGorsellestirici

__all__ = [
    "ModernArtirici",
    "YumusatilmisCrossEntropyLoss",
    "ModernRegulerVisionNet",
    "RegulerizasyonLaboratuvari",
    "RegulerizasyonGorsellestirici"
]
