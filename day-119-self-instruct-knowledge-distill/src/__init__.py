"""
Day 119: Knowledge Distillation & Self-Instruct Paketi.
"""

from .damitma_kaybi import KnowledgeDistillationLoss
from .ogretmen_ogrenci_modeller import (
    TransformerLM,
    ogretmen_model_uret,
    ogrenci_model_uret,
)
from .self_instruct_ureteci import SelfInstructUreteci
from .damitma_laboratuvari import DamitmaLaboratuvari
from .gorsellestirici import DamitmaGorsellestirici

__all__ = [
    "KnowledgeDistillationLoss",
    "TransformerLM",
    "ogretmen_model_uret",
    "ogrenci_model_uret",
    "SelfInstructUreteci",
    "DamitmaLaboratuvari",
    "DamitmaGorsellestirici",
]
