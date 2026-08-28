"""
Hedef Değişken ve Ön İşleme Sızıntı Dedektörü (Target & Preprocessing Leakage Detector).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


class TargetLeakageDedektoru:
    """Veri setindeki şüpheli hedef sızıntılarını (Target Leakage) ve yapay yüksek korelasyonları tespit eder."""

    def __init__(self, korelasyon_esigi: float = 0.88):
        self.korelasyon_esigi = korelasyon_esigi

    def denetle(self, df: pd.DataFrame, hedef_kolon: str) -> Dict[str, Any]:
        """Tüm öznitelikleri hedef değişkenle olan ilişkisine göre tarar ve risk raporu üretir."""
        if hedef_kolon not in df.columns:
            return {"durum": "HATA", "mesaj": f"Hedef kolon '{hedef_kolon}' bulunamadı."}

        supheli_kolonlar = []
        korelasyonlar = {}
        y = df[hedef_kolon].values

        sayisal_kolonlar = df.select_dtypes(include=[np.number]).columns

        for col in sayisal_kolonlar:
            if col == hedef_kolon:
                continue

            x = df[col].fillna(df[col].median()).values
            if len(np.unique(x)) > 1:
                r = float(np.corrcoef(x, y)[0, 1])
                korelasyonlar[col] = float(round(r, 4))

                if abs(r) >= self.korelasyon_esigi:
                    supheli_kolonlar.append({
                        "kolon": col,
                        "korelasyon": float(round(r, 4)),
                        "sebep": f"Hedefle aşırı yüksek korelasyon (|r| >= {self.korelasyon_esigi}). Target leakage şüphesi!",
                        "oneri": "Bu özniteliği model eğitim setinden çıkarın veya zaman damgası kontrolü yapın."
                    })

        sizinti_var = len(supheli_kolonlar) > 0

        return {
            "durum": "SIZINTI_RISKI_TESPIT_EDILDI" if sizinti_var else "GUVENLI_VERI_SETI",
            "toplam_incelenen_sayisal_kolon": len(korelasyonlar),
            "supheli_kolon_sayisi": len(supheli_kolonlar),
            "supheli_kolonlar": supheli_kolonlar,
            "tum_korelasyonlar": korelasyonlar
        }
