"""
Spekülatif Çıkarım Hızlanma ve Kabul Oranı Profilleyicisi (Day 193 - FAZ 10).
Leviathan Teoremi: Beklenen Kabul Sayısı, Tepe Hızlanma Katsayısı ve Donanım Analitiği.
"""

from typing import Dict, Any, List
import numpy as np


class SpekulatifHizProfilleyici:
    """
    Spekülatif Çıkarım Matematiksel Hız ve Kabul Analiz Motoru.
    """

    @classmethod
    def teorik_hizlanma_analizi(
        cls,
        kabul_orani: float = 0.80,  # alpha (Kabul Oranı)
        gamma: int = 4,             # K (Önerilen Token Sayısı)
        draft_target_maliyet_orani: float = 0.08,  # c = T_draft / T_target
    ) -> Dict[str, Any]:
        """Leviathan et al. teorik hızlanma formülü hesabı."""
        # Beklenen adım başına kabul edilen token: E = (1 - alpha^(K+1)) / (1 - alpha)
        if abs(kabul_orani - 1.0) < 1e-4:
            beklenen_token = float(gamma + 1)
        else:
            beklenen_token = (1.0 - (kabul_orani ** (gamma + 1))) / (1.0 - kabul_orani)

        # Adım maliyeti: 1 (Target Forward) + gamma * c (Draft Forward)
        adim_maliyeti = 1.0 + (gamma * draft_target_maliyet_orani)
        teorik_hizlanma = beklenen_token / adim_maliyeti

        return {
            "kabul_orani": kabul_orani,
            "gamma_k": gamma,
            "draft_maliyet_orani": draft_target_maliyet_orani,
            "beklenen_token_adim_basi": round(beklenen_token, 2),
            "adim_maliyeti": round(adim_maliyeti, 2),
            "teorik_hizlanma": round(teorik_hizlanma, 2),
            "hizlanma_faktoru": f"{teorik_hizlanma:.2f}x",
        }

    @classmethod
    def parametre_duyarlilik_tarama_raporu(cls) -> List[Dict[str, Any]]:
        """Farklı kabul oranları (0.50 - 0.95) için hızlanma tablosu."""
        oranlar = [0.50, 0.65, 0.75, 0.85, 0.95]
        rapor = []
        for a in oranlar:
            p = cls.teorik_hizlanma_analizi(kabul_orani=a, gamma=4, draft_target_maliyet_orani=0.08)
            rapor.append({
                "kabul_orani": a,
                "gamma_k": 4,
                "beklenen_token": p["beklenen_token_adim_basi"],
                "hizlanma": p["hizlanma_faktoru"],
            })
        return rapor
