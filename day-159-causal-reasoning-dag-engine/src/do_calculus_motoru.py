"""
Do-Calculus ve Müdahale (Intervention) Motoru (Day 159 - Faz 8).
Backdoor kriteri uygulayarak sahte korelasyonu temizler ve gerçek nedensel etkiyi (ATE) hesaplar.
"""

from typing import Dict, Any
from .nedensel_dag_modeli import NedenselDAGModeli


class DoCalculusMotoru:
    """Müdahale ve Do-Calculus operatörü."""

    @classmethod
    def mudahale_etkisi_hesapla(cls, model: NedenselDAGModeli) -> Dict[str, float]:
        """
        2. Basamak: P(Y=1 | do(X=1)) ve P(Y=1 | do(X=0)) Backdoor Adjustment.
        """
        # P(Y=1 | do(X=1)) = P(Y=1|X=1, Z=0)*P(Z=0) + P(Y=1|X=1, Z=1)*P(Z=1)
        p_y1_do_x1 = (
            model.p_y_given_xz[(1, 0)] * model.p_z[0] +
            model.p_y_given_xz[(1, 1)] * model.p_z[1]
        )

        # P(Y=1 | do(X=0)) = P(Y=1|X=0, Z=0)*P(Z=0) + P(Y=1|X=0, Z=1)*P(Z=1)
        p_y1_do_x0 = (
            model.p_y_given_xz[(0, 0)] * model.p_z[0] +
            model.p_y_given_xz[(0, 1)] * model.p_z[1]
        )

        ate = p_y1_do_x1 - p_y1_do_x0  # Average Treatment Effect

        return {
            "p_y1_do_x1": round(p_y1_do_x1, 3),  # 0.700
            "p_y1_do_x0": round(p_y1_do_x0, 3),  # 0.600
            "ortalama_nedensel_etki_ate": round(ate, 3),  # +0.100 (%10)
        }
