"""
Z3 SMT Çözücü Modülü (Day 150 - Faz 8).
Mantıksal kısıt sağlama (SMT / SAT) ve birinci dereceden mantık (First-Order Logic) ispat motoru.
"""

from typing import Dict, Any, Optional
import z3


class Z3SMTCozucu:
    """Z3 SMT (Satisfiability Modulo Theories) kısıt çözücü motoru."""

    @classmethod
    def sopa_ve_top_coz(cls) -> Dict[str, Any]:
        """
        Z3 ile Sopa & Top problemini çözer:
        Sopa + Top == 1.10
        Sopa == Top + 1.00
        """
        solver = z3.Solver()
        sopa = z3.Real("sopa")
        top = z3.Real("top")

        solver.add(sopa + top == z3.RealVal("1.10"))
        solver.add(sopa == top + z3.RealVal("1.00"))
        solver.add(top > 0)

        durum = solver.check()
        if durum == z3.sat:
            model = solver.model()
            top_val = float(model[top].as_decimal(4))
            sopa_val = float(model[sopa].as_decimal(4))
            return {
                "sat_mi": True,
                "top": top_val,
                "sopa": sopa_val,
                "durum_metni": "SAT (Tatmin Edilebilir)",
            }
        else:
            return {"sat_mi": False, "durum_metni": "UNSAT"}

    @classmethod
    def tam_sayi_kisit_coz(cls, toplam: int = 15, carpim: int = 56) -> Dict[str, Any]:
        """
        x + y == toplam
        x * y == carpim
        x, y > 0
        """
        solver = z3.Solver()
        x = z3.Int("x")
        y = z3.Int("y")

        solver.add(x > 0, y > 0)
        solver.add(x + y == toplam)
        solver.add(x * y == carpim)

        durum = solver.check()
        if durum == z3.sat:
            model = solver.model()
            return {
                "sat_mi": True,
                "x": model[x].as_long(),
                "y": model[y].as_long(),
                "durum_metni": "SAT",
            }
        else:
            return {"sat_mi": False, "durum_metni": "UNSAT"}
