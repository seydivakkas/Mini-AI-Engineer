"""
Sim2Real Domain Randomization Başarım ve Dayanıklılık Profilleyicisi (Day 247).
Naive Sim vs Visual DR vs Dynamics DR vs Full Multimodal DR Analizi.
"""

from typing import Dict, Any, List
from .domain_randomization_motoru import (
    VisualRandomizer,
    DynamicsRandomizer,
    ActionDelayInjector,
    Sim2RealEvaluator,
)


class Sim2RealProfilleyici:
    """FAZ 13 Sim2Real ve Domain Randomization Profilleyicisi."""

    @classmethod
    def basarim_profili_cikar(cls) -> Dict[str, Any]:
        """4 Rejim Kıyaslama Raporu ve Dayanıklılık Metrikleri."""
        rejimler = ["naive_sim", "visual_dr", "dynamics_dr", "full_multimodal_dr"]
        sonuclar = {r: Sim2RealEvaluator.evaluate_regime(r) for r in rejimler}

        karsilastirma = {
            "gercek_dunya_basari_yuzdesi": {
                "Naive_Sim": sonuclar["naive_sim"]["basari_orani_yuzde"],
                "Visual_DR": sonuclar["visual_dr"]["basari_orani_yuzde"],
                "Dynamics_DR": sonuclar["dynamics_dr"]["basari_orani_yuzde"],
                "Full_Multimodal_DR": sonuclar["full_multimodal_dr"]["basari_orani_yuzde"],
            },
            "ortalama_yorunge_hatasi_cm": {
                "Naive_Sim": sonuclar["naive_sim"]["ortalama_hata_cm"],
                "Visual_DR": sonuclar["visual_dr"]["ortalama_hata_cm"],
                "Dynamics_DR": sonuclar["dynamics_dr"]["ortalama_hata_cm"],
                "Full_Multimodal_DR": sonuclar["full_multimodal_dr"]["ortalama_hata_cm"],
            },
            "motor_tork_asimi_yuzdesi": {
                "Naive_Sim": sonuclar["naive_sim"]["tork_asimi_yuzdesi"],
                "Visual_DR": sonuclar["visual_dr"]["tork_asimi_yuzdesi"],
                "Dynamics_DR": sonuclar["dynamics_dr"]["tork_asimi_yuzdesi"],
                "Full_Multimodal_DR": sonuclar["full_multimodal_dr"]["tork_asimi_yuzdesi"],
            },
            "zero_shot_robust_skoru": {
                "Naive_Sim": sonuclar["naive_sim"]["zero_shot_robust_skor"],
                "Visual_DR": sonuclar["visual_dr"]["zero_shot_robust_skor"],
                "Dynamics_DR": sonuclar["dynamics_dr"]["zero_shot_robust_skor"],
                "Full_Multimodal_DR": sonuclar["full_multimodal_dr"]["zero_shot_robust_skor"],
            },
        }

        ornek_dinamik = DynamicsRandomizer.sample_dynamics_parameters(tohum=10)

        return {
            "karsilastirma": karsilastirma,
            "ornek_dinamik_parametreler": ornek_dinamik,
            "rejim_sayisi": len(rejimler),
        }
