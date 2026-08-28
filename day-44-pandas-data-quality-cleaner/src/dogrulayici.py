"""
Pandas Tablo Şema ve Veri Kalitesi Doğrulama Motoru (Schema Validator).
"""

from typing import Dict, Any, List
import time
import numpy as np
import pandas as pd
from .sema import TabloSemasi, KolonKurali


class SemaDogrulayici:
    """Veri çerçevesini (DataFrame) tanımlı şema kurallarına göre denetler ve kalite skoru üretir."""

    def __init__(self, sema: TabloSemasi):
        self.sema = sema

    def dogrula(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tüm şema kurallarını ve veri kalitesi kriterlerini çalıştırır."""
        baslangic = time.perf_counter()
        ihlaller = []
        kolon_metrikleri = {}
        toplam_hucre = int(df.shape[0] * df.shape[1]) if df.shape[0] > 0 else 1
        hatali_hucre_sayisi = 0

        # 1. Kolon Varlık ve Beklenmeyen Kolon Kontrolü
        mevcut_kolonlar = set(df.columns)
        beklenen_kolonlar = set(self.sema.kolon_kurallari.keys())

        eksik_kolonlar = beklenen_kolonlar - mevcut_kolonlar
        for ek in eksik_kolonlar:
            ihlaller.append({
                "kolon": ek,
                "kod": "EKSIK_ZORUNLU_KOLON",
                "mesaj": f"'{ek}' isimli zorunlu kolon veri çerçevesinde bulunamadı.",
                "kritik": True,
                "etkilenen_satir": len(df)
            })

        fazla_kolonlar = mevcut_kolonlar - beklenen_kolonlar
        if self.sema.beklenmeyen_kolon_engeli and len(fazla_kolonlar) > 0:
            ihlaller.append({
                "kolon": ", ".join(fazla_kolonlar),
                "kod": "BEKLENMEYEN_KOLON",
                "mesaj": f"Şemada tanımlanmayan fazlalık kolonlar tespit edildi: {fazla_kolonlar}",
                "kritik": False,
                "etkilenen_satir": 0
            })

        # 2. Çift Satır (Duplicate Rows) Kontrolü
        cift_satir_sayisi = int(df.duplicated().sum())
        cift_orani = float(cift_satir_sayisi / max(len(df), 1))
        if cift_orani > self.sema.izin_verilen_cift_satir_orani:
            ihlaller.append({
                "kolon": "_ALL_",
                "kod": "CIFT_SATIR_IHLALI",
                "mesaj": f"{cift_satir_sayisi} adet mükerrer (duplicate) satır tespit edildi (%{cift_orani*100:.2f}).",
                "kritik": False,
                "etkilenen_satir": cift_satir_sayisi
            })
            hatali_hucre_sayisi += cift_satir_sayisi * len(df.columns)

        # 3. Kolon Bazlı Detaylı Denetim
        for kolon_ad, kural in self.sema.kolon_kurallari.items():
            if kolon_ad not in df.columns:
                continue

            seri = df[kolon_ad]
            n_satir = len(seri)
            k_hatali_satir = 0

            # A. Null / Eksik Değer Kontrolü
            null_sayisi = int(seri.isna().sum())
            null_orani = float(null_sayisi / max(n_satir, 1))

            if null_orani > kural.izin_verilen_null_orani:
                ihlaller.append({
                    "kolon": kolon_ad,
                    "kod": "ASIRI_NULL_ORANI",
                    "mesaj": f"'{kolon_ad}' kolonunda {null_sayisi} adet eksik değer var (%{null_orani*100:.1f} > %{kural.izin_verilen_null_orani*100:.1f}).",
                    "kritik": null_orani > 0.50,
                    "etkilenen_satir": null_sayisi
                })
                k_hatali_satir += null_sayisi

            # B. Benzersizlik (Uniqueness) Kontrolü
            if kural.benzersiz:
                cift_id_sayisi = int(seri.dropna().duplicated().sum())
                if cift_id_sayisi > 0:
                    ihlaller.append({
                        "kolon": kolon_ad,
                        "kod": "BENZERSIZLIK_IHLALI",
                        "mesaj": f"'{kolon_ad}' kolonunda {cift_id_sayisi} adet tekil olmayan (duplicate ID) kayıt bulundu.",
                        "kritik": True,
                        "etkilenen_satir": cift_id_sayisi
                    })
                    k_hatali_satir += cift_id_sayisi

            dolu_seri = seri.dropna()

            # C. Değer Aralığı Kontrolü (Numeric Range)
            if kural.min_deger is not None or kural.max_deger is not None:
                numeric_seri = pd.to_numeric(dolu_seri, errors="coerce")
                aralik_disi_mask = pd.Series(False, index=dolu_seri.index)
                if kural.min_deger is not None:
                    aralik_disi_mask |= (numeric_seri < kural.min_deger)
                if kural.max_deger is not None:
                    aralik_disi_mask |= (numeric_seri > kural.max_deger)

                aralik_disi_sayi = int(aralik_disi_mask.sum())
                if aralik_disi_sayi > 0:
                    ihlaller.append({
                        "kolon": kolon_ad,
                        "kod": "ARALIK_DISI_DEGER",
                        "mesaj": f"'{kolon_ad}' kolonunda {aralik_disi_sayi} adet değer [{kural.min_deger}, {kural.max_deger}] sınırları dışında.",
                        "kritik": False,
                        "etkilenen_satir": aralik_disi_sayi
                    })
                    k_hatali_satir += aralik_disi_sayi

            # D. Kategorik Küme Kontrolü (Categorical Domain)
            if kural.kategoriler is not None:
                gecersiz_kat_mask = ~dolu_seri.astype(str).isin(kural.kategoriler)
                gecersiz_kat_sayi = int(gecersiz_kat_mask.sum())
                if gecersiz_kat_sayi > 0:
                    ihlaller.append({
                        "kolon": kolon_ad,
                        "kod": "GECERSIZ_KATEGORI",
                        "mesaj": f"'{kolon_ad}' kolonunda {gecersiz_kat_sayi} adet izin verilmeyen kategori değeri bulundu.",
                        "kritik": False,
                        "etkilenen_satir": gecersiz_kat_sayi
                    })
                    k_hatali_satir += gecersiz_kat_sayi

            # E. RegEx Kalıp Kontrolü (Pattern Match)
            if kural.regex_kalibi is not None:
                gecersiz_regex_mask = ~dolu_seri.astype(str).str.match(kural.regex_kalibi)
                gecersiz_regex_sayi = int(gecersiz_regex_mask.sum())
                if gecersiz_regex_sayi > 0:
                    ihlaller.append({
                        "kolon": kolon_ad,
                        "kod": "REGEX_DESEN_IHLALI",
                        "mesaj": f"'{kolon_ad}' kolonunda {gecersiz_regex_sayi} adet değer beklenen format kalıbına uymuyor.",
                        "kritik": False,
                        "etkilenen_satir": gecersiz_regex_sayi
                    })
                    k_hatali_satir += gecersiz_regex_sayi

            hatali_hucre_sayisi += k_hatali_satir
            kolon_metrikleri[kolon_ad] = {
                "toplam_satir": n_satir,
                "null_sayisi": null_sayisi,
                "null_orani": float(round(null_orani * 100.0, 2)),
                "hatali_deger_sayisi": k_hatali_satir,
                "kalite_orani": float(round(max(0.0, 100.0 - (k_hatali_satir / max(n_satir, 1) * 100.0)), 2))
            }

        # 4. Kalite Skoru ve Genel Karar
        sure_ms = float((time.perf_counter() - baslangic) * 1000.0)
        kritik_sayisi = sum(1 for ih in ihlaller if ih["kritik"])

        kalite_skoru = float(max(0.0, min(100.0, 100.0 - (hatali_hucre_sayisi / toplam_hucre * 100.0))))

        if len(ihlaller) == 0:
            karar = "GECERLI_MUKEMMEL"
            durum = "TABLO_ONAYLANDI"
        elif kritik_sayisi > 0 or kalite_skoru < 60.0:
            karar = "KRITIK_RED"
            durum = "TABLO_REDDEDILDI_ISLENEMEZ"
        else:
            karar = "DUZELTILEBILIR_KIRLI_VERI"
            durum = "OTOMATIK_TEMIZLEME_GEREKLI"

        return {
            "tablo_adi": self.sema.tablo_adi,
            "karar": karar,
            "durum": durum,
            "kalite_skoru": float(round(kalite_skoru, 2)),
            "denetim_suresi_ms": float(round(sure_ms, 3)),
            "satir_sayisi": len(df),
            "sutun_sayisi": len(df.columns),
            "toplam_ihlal_sayisi": len(ihlaller),
            "kritik_ihlal_sayisi": kritik_sayisi,
            "cift_satir_sayisi": cift_satir_sayisi,
            "kolon_raporlari": kolon_metrikleri,
            "ihlaller": ihlaller
        }
