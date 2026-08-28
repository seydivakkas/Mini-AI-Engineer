"""
Otomatik Tabüler Veri Temizleme ve İmpütasyon Motoru (Automated Data Quality Cleaner).
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from .sema import TabloSemasi


class OtomatikVeriTemizleyici:
    """Doğrulama ihlali alan tabloları otomatik düzelterek temiz ve üretime hazır hale getirir."""

    def __init__(self, sema: TabloSemasi):
        self.sema = sema

    def temizle_ve_iyilestir(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Veri çerçevesi üzerinde sıralı veri hijyeni adımlarını uygular."""
        temiz_df = df.copy()
        yapilan_islemler = []
        baslangic_satir = len(temiz_df)

        # 1. Çift Satırların Temizlenmesi (Deduplication)
        cift_sayisi = int(temiz_df.duplicated().sum())
        if cift_sayisi > 0:
            temiz_df = temiz_df.drop_duplicates().reset_index(drop=True)
            yapilan_islemler.append(f"MUKERRER_SATIRLAR_SILINDI_{cift_sayisi}_ADET")

        # 2. Kolon Bazlı İmpütasyon ve Sınır Kırpma
        for kolon_ad, kural in self.sema.kolon_kurallari.items():
            if kolon_ad not in temiz_df.columns:
                continue

            # A. Tip Dönüşümü (Type Coercion)
            if kural.tip in [int, float, np.int64, np.float64, "int", "float"]:
                temiz_df[kolon_ad] = pd.to_numeric(temiz_df[kolon_ad], errors="coerce")

            # B. Null Değer İmpütasyonu (Eksik Veri Doldurma)
            null_sayisi = int(temiz_df[kolon_ad].isna().sum())
            if null_sayisi > 0:
                if kural.benzersiz:
                    # Benzersiz ID'lerde null olan satırları düşür
                    temiz_df = temiz_df.dropna(subset=[kolon_ad]).reset_index(drop=True)
                    yapilan_islemler.append(f"BENZERSIZ_KOLON_NULL_SATIRLAR_SILINDI_{kolon_ad}")
                elif kural.varsayilan_doldurma == "median":
                    medyan_val = temiz_df[kolon_ad].median()
                    temiz_df[kolon_ad] = temiz_df[kolon_ad].fillna(medyan_val)
                    yapilan_islemler.append(f"NULL_DOLDURULDU_MEDYAN_{kolon_ad}_{medyan_val:.2f}")
                elif kural.varsayilan_doldurma == "mean":
                    ort_val = temiz_df[kolon_ad].mean()
                    temiz_df[kolon_ad] = temiz_df[kolon_ad].fillna(ort_val)
                    yapilan_islemler.append(f"NULL_DOLDURULDU_ORTALAMA_{kolon_ad}_{ort_val:.2f}")
                elif kural.varsayilan_doldurma == "mode":
                    mod_seri = temiz_df[kolon_ad].mode()
                    mod_val = mod_seri.iloc[0] if len(mod_seri) > 0 else "BILINMIYOR"
                    temiz_df[kolon_ad] = temiz_df[kolon_ad].fillna(mod_val)
                    yapilan_islemler.append(f"NULL_DOLDURULDU_MOD_{kolon_ad}_{mod_val}")
                elif kural.varsayilan_doldurma is not None:
                    temiz_df[kolon_ad] = temiz_df[kolon_ad].fillna(kural.varsayilan_doldurma)
                    yapilan_islemler.append(f"NULL_DOLDURULDU_SABIT_{kolon_ad}_{kural.varsayilan_doldurma}")

            # C. Değer Aralığı Kırpma (Clamping)
            if kural.min_deger is not None or kural.max_deger is not None:
                temiz_df[kolon_ad] = temiz_df[kolon_ad].clip(lower=kural.min_deger, upper=kural.max_deger)
                yapilan_islemler.append(f"SINIR_KIRPILDI_CLIP_{kolon_ad}_[{kural.min_deger},{kural.max_deger}]")

            # D. Kategorik Değer Temizliği ve Boşluk Budama
            if kural.kategoriler is not None:
                temiz_df[kolon_ad] = temiz_df[kolon_ad].astype(str).str.strip().str.upper()
                # Geçersiz kategorileri en çok tekrar edenle doldur
                gecersiz_mask = ~temiz_df[kolon_ad].isin(kural.kategoriler)
                if gecersiz_mask.sum() > 0:
                    gecerli_mod = kural.kategoriler[0]
                    temiz_df.loc[gecersiz_mask, kolon_ad] = gecerli_mod
                    yapilan_islemler.append(f"GECERSIZ_KATEGORILER_DUZELTILDİ_{kolon_ad}_{gecerli_mod}")

        bitis_satir = len(temiz_df)

        return temiz_df, {
            "baslangic_satir_sayisi": baslangic_satir,
            "temizlenmis_satir_sayisi": bitis_satir,
            "silinen_satir_sayisi": baslangic_satir - bitis_satir,
            "yapilan_islemler": yapilan_islemler
        }
