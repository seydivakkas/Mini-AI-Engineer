"""
Tabüler Veri Şeması ve Deklaratif Doğrulama Kuralları (Pandera / Great Expectations Benzeri).
"""

from typing import Dict, List, Optional, Any, Union
import re
import numpy as np
import pandas as pd


class KolonKurali:
    """Tek bir tabüler sütunun beklenen tip, aralık ve desen kuralları."""

    def __init__(
        self,
        ad: str,
        tip: Union[str, type],
        zorunlu: bool = True,
        benzersiz: bool = False,
        min_deger: Optional[float] = None,
        max_deger: Optional[float] = None,
        kategoriler: Optional[List[str]] = None,
        regex_kalibi: Optional[str] = None,
        izin_verilen_null_orani: float = 0.0,
        varsayilan_doldurma: Optional[Any] = "median"
    ):
        self.ad = ad
        self.tip = tip
        self.zorunlu = zorunlu
        self.benzersiz = benzersiz
        self.min_deger = min_deger
        self.max_deger = max_deger
        self.kategoriler = kategoriler
        self.regex_kalibi = re.compile(regex_kalibi) if regex_kalibi else None
        self.izin_verilen_null_orani = izin_verilen_null_orani
        self.varsayilan_doldurma = varsayilan_doldurma


class TabloSemasi:
    """Veri tablosunun deklaratif sözleşmesi (Schema Contract)."""

    def __init__(
        self,
        tablo_adi: str = "UretimVeriSeti",
        kolon_kurallari: Optional[List[KolonKurali]] = None,
        beklenmeyen_kolon_engeli: bool = False,
        izin_verilen_cift_satir_orani: float = 0.0
    ):
        self.tablo_adi = tablo_adi
        self.kolon_kurallari: Dict[str, KolonKurali] = {k.ad: k for k in (kolon_kurallari or [])}
        self.beklenmeyen_kolon_engeli = beklenmeyen_kolon_engeli
        self.izin_verilen_cift_satir_orani = izin_verilen_cift_satir_orani

    def kolon_ekle(self, kural: KolonKurali) -> "TabloSemasi":
        self.kolon_kurallari[kural.ad] = kural
        return self
