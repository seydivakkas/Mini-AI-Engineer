"""
Plan-and-Solve Araçları Modülü (Day 122 - Faz 7).
Alt görevlerin icrasında kullanılan matematiksel ve anlamsal işlem araçları.
"""

from typing import Dict, Any
import ast
import operator
import re


class AritmetikHesaplayici:
    """Alt görevlerdeki matematiksel formülleri değişken ikamesiyle hesaplar."""

    ISLEMLER = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return self.ISLEMLER[type(node.op)](self._eval(node.left), self._eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return -self._eval(node.operand)
        raise ValueError(f"Desteklenmeyen düğüm: {type(node)}")

    def hesapla(self, ifade: str, degiskenler: Dict[str, float]) -> float:
        """İfadedeki değişkenleri sözlükteki değerlerle ikame edip hesaplar."""
        islenmis = ifade
        for k, v in degiskenler.items():
            islenmis = re.sub(rf"\b{k}\b", str(v), islenmis)

        tree = ast.parse(islenmis.replace(" ", ""), mode="eval")
        return float(self._eval(tree.body))


class VeriCikarici:
    """Metin içerisinden sayısal değişkenleri ve sabitleri çıkarır."""

    def cikar(self, metin: str) -> Dict[str, float]:
        bulunanlar: Dict[str, float] = {}
        # Sayı ve birim eşleştirmesi
        sayilar = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", metin)
        for idx, s in enumerate(sayilar, start=1):
            bulunanlar[f"x_{idx}"] = float(s)
        return bulunanlar


class MetinBirlestirici:
    """Alt görevlerin sonuçlarını mantıksal bir nihai yanıtta birleştirir."""

    def sentezle(self, ana_gorev: str, sonuclar: Dict[str, Any]) -> str:
        parcalar = [f"Görev: '{ana_gorev}' için adım adım çözüm tamamlandı:"]
        for g_id, sonuc in sonuclar.items():
            parcalar.append(f"  • {g_id}: {sonuc}")
        return "\n".join(parcalar)
