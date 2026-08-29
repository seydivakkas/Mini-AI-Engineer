"""
SymPy Sembolik Matematik Motoru (Day 150 - Faz 8).
Cebirsel denklem çözümü, sembolik sadeleştirme, türev, integral ve modüler aritmetik motoru.
"""

from typing import List, Any, Dict
import sympy as sp


class SymPySembolikMotor:
    """Sembolik matematiksel ispat ve kesin hesaplama motoru."""

    @classmethod
    def denklem_coz(cls, ifade_sol: str, ifade_sag: str = "0", degisken_str: str = "x") -> List[Any]:
        """
        ifade_sol = ifade_sag denklemini sembolik olarak çözer.
        Örn: x**2 - 5*x + 6 = 0 => [2, 3]
        """
        x = sp.Symbol(degisken_str)
        sol = sp.sympify(ifade_sol)
        sag = sp.sympify(ifade_sag)
        denklem = sp.Eq(sol, sag)
        kokler = sp.solve(denklem, x)
        return [float(k) if k.is_real and k.is_number else str(k) for k in kokler]

    @classmethod
    def moduler_coz(cls, a: int, b: int, m: int) -> int:
        """
        a * x = b (mod m) kongrüansını çözer.
        Örn: 3x + 7 = 2 (mod 5) => 3x = 0 (mod 5) => x = 0
        """
        # sp.divmod / gcdex
        x_cozum = None
        for aday in range(m):
            if (a * aday) % m == (b % m):
                x_cozum = aday
                break
        if x_cozum is None:
            raise ValueError(f"{a}x = {b} (mod {m}) için çözüm bulunamadı!")
        return x_cozum

    @classmethod
    def turev_al(cls, ifade_str: str, degisken_str: str = "x") -> str:
        """Sembolik türev alır: d/dx (f(x))."""
        x = sp.Symbol(degisken_str)
        f = sp.sympify(ifade_str)
        turev = sp.diff(f, x)
        return str(turev)

    @classmethod
    def integral_al(cls, ifade_str: str, degisken_str: str = "x") -> str:
        """Sembolik integral alır: int f(x) dx."""
        x = sp.Symbol(degisken_str)
        f = sp.sympify(ifade_str)
        integral = sp.integrate(f, x)
        return str(integral)

    @classmethod
    def sadelestir(cls, ifade_str: str) -> str:
        """Sembolik ifadeyi en sade biçimine indirger."""
        f = sp.sympify(ifade_str)
        return str(sp.simplify(f))
