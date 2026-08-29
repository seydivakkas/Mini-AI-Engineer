"""
Rollout / Simülasyon Politika Motoru (Day 146 - Faz 8).
MCTS simülasyon adımında terminal duruma kadar hızlı sezgisel rollout ve ödül tahmini.
"""

from typing import List


class RolloutPolitikaMotoru:
    """MCTS simülasyon aşamasında durumları terminale kadar koşan politika motoru."""

    @classmethod
    def can_solve(cls, sayilar: List[float], hedef: float = 24.0) -> bool:
        """Kalan sayılardan hedefe ulaşılabiliyor mu (Derinlik kısıtlı çözüm kontrolü)."""
        if len(sayilar) == 1:
            return abs(sayilar[0] - hedef) < 1e-4

        n = len(sayilar)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sayilar[i], sayilar[j]
                digerleri = [sayilar[k] for k in range(n) if k != i and k != j]

                ops = [a + b, a - b, b - a, a * b]
                if abs(b) > 1e-5:
                    ops.append(a / b)
                if abs(a) > 1e-5:
                    ops.append(b / a)

                for r in ops:
                    if cls.can_solve(digerleri + [r], hedef):
                        return True
        return False

    @classmethod
    def simule_et(cls, sayilar: List[float], hedef: float = 24.0) -> float:
        """
        Durumu simüle eder: Hedefe ulaşılabiliyorsa 1.0, ulaşılamıyorsa 0.0 döner.
        """
        if cls.can_solve(sayilar, hedef):
            return 1.0
        return 0.0
