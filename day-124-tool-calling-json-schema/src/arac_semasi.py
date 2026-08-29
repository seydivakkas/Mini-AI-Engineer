"""
JSON Schema ve Kesin Tip Güvenliği Modülü (Day 124 - Faz 7).
OpenAI/Anthropic uyumlu araç şemaları, Pydantic benzeri doğrulama ve tip denetimi.
"""

from typing import Dict, Any, List, Optional, Tuple, Union


class ParametreOzelligi:
    """Tek bir parametrenin JSON Schema özelliklerini tanımlar."""

    def __init__(
        self,
        tip: str,
        aciklama: str,
        enum: Optional[List[Any]] = None,
        minimum: Optional[Union[int, float]] = None,
        maximum: Optional[Union[int, float]] = None,
    ):
        self.tip = tip  # "string", "integer", "number", "boolean", "array"
        self.aciklama = aciklama
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

    def to_json_schema(self) -> Dict[str, Any]:
        sema: Dict[str, Any] = {"type": self.tip, "description": self.aciklama}
        if self.enum is not None:
            sema["enum"] = self.enum
        if self.minimum is not None:
            sema["minimum"] = self.minimum
        if self.maximum is not None:
            sema["maximum"] = self.maximum
        return sema


class AracSemasi:
    """OpenAI/Anthropic standardında araç fonksiyonu şeması ve validatörü."""

    def __init__(
        self,
        ad: str,
        aciklama: str,
        parametreler: Dict[str, ParametreOzelligi],
        zorunlu_alanlar: Optional[List[str]] = None,
    ):
        self.ad = ad
        self.aciklama = aciklama
        self.parametreler = parametreler
        self.zorunlu_alanlar = zorunlu_alanlar or []

    def to_openai_schema(self) -> Dict[str, Any]:
        """OpenAI standardında function calling JSON Schema sözlüğü üretir."""
        ozellikler = {k: v.to_json_schema() for k, v in self.parametreler.items()}
        return {
            "type": "function",
            "function": {
                "name": self.ad,
                "description": self.aciklama,
                "parameters": {
                    "type": "object",
                    "properties": ozellikler,
                    "required": self.zorunlu_alanlar,
                },
            },
        }

    def dogrula(self, argumanlar: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Gelen argümanları şema kurallarına göre kesin doğrular."""
        hatalar: List[str] = []

        if not isinstance(argumanlar, dict):
            return False, ["Argümanlar geçerli bir JSON nesnesi (dict) olmalıdır."]

        # 1. Zorunlu Alan Kontrolü
        for z in self.zorunlu_alanlar:
            if z not in argumanlar:
                hatalar.append(f"Zorunlu alan eksik: '{z}'")

        # 2. Tip ve Aralık Kontrolleri
        for anahtar, deger in argumanlar.items():
            if anahtar not in self.parametreler:
                hatalar.append(f"Bilinmeyen argüman: '{anahtar}'")
                continue

            param = self.parametreler[anahtar]

            # Tip Uyumu Denetimi
            if param.tip == "string" and not isinstance(deger, str):
                hatalar.append(f"'{anahtar}' string olmalıdır, {type(deger).__name__} verildi.")
            elif param.tip == "integer" and (not isinstance(deger, int) or isinstance(deger, bool)):
                hatalar.append(f"'{anahtar}' integer olmalıdır, {type(deger).__name__} verildi.")
            elif param.tip == "number" and not isinstance(deger, (int, float)):
                hatalar.append(f"'{anahtar}' sayı (number) olmalıdır, {type(deger).__name__} verildi.")
            elif param.tip == "boolean" and not isinstance(deger, bool):
                hatalar.append(f"'{anahtar}' boolean olmalıdır, {type(deger).__name__} verildi.")

            # Enum Uyumu Denetimi
            if param.enum is not None and deger not in param.enum:
                hatalar.append(f"'{anahtar}' değeri ({deger}) izin verilen listede yok: {param.enum}")

            # Sayısal Aralık Denetimi
            if isinstance(deger, (int, float)):
                if param.minimum is not None and deger < param.minimum:
                    hatalar.append(f"'{anahtar}' minimum {param.minimum} olmalıdır (Verilen: {deger})")
                if param.maximum is not None and deger > param.maximum:
                    hatalar.append(f"'{anahtar}' maksimum {param.maximum} olmalıdır (Verilen: {deger})")

        return len(hatalar) == 0, hatalar
