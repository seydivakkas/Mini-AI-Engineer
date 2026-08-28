"""
Day 47: Scikit-Learn ile Veri Sızıntısına Karşı Güvenli Pipeline Ana Yürütme Betiği.
"""

import os
import numpy as np
import pandas as pd
from src.pipeline_mimari import GuvenliPipelineUretici
from src.sizinti_dedektoru import TargetLeakageDedektoru
from src.nested_cv_motoru import NestedCVMotoru
from src.gorsellestirici import PipelineTehisGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 47: SCIKIT-LEARN İLE VERİ SIZINTISINA (DATA LEAKAGE) KARŞI GÜVENLİ PIPELINE")
    print("=" * 85)

    # 1. Gerçekçi Tabüler Kredi Risk Veri Setinin Simülasyonu
    np.random.seed(42)
    n = 600

    gelir = np.random.lognormal(mean=9.5, sigma=0.6, size=n)
    kredi = np.random.uniform(5000, 150000, size=n)
    yas = np.random.randint(20, 65, size=n)
    harcama = np.random.uniform(100, 15000, size=n)

    meslek = np.random.choice(["MUHENDIS", "DOKTOR", "MEMUR", "SERBEST", "OGRENCI"], size=n)
    segment = np.random.choice(["A", "B", "C", "D"], size=n)
    sehir = np.random.choice(["ISTANBUL", "ANKARA", "IZMIR", "BURSA"], size=n)

    # Hedef Değişken (0: Red, 1: Onay)
    logit = (gelir / 20000.0) - (kredi / 50000.0) + (yas / 30.0) - (harcama / 8000.0)
    prob = 1.0 / (1.0 + np.exp(-logit))
    hedef = (prob >= np.median(prob)).astype(int)

    df = pd.DataFrame({
        "gelir": gelir,
        "kredi_tutari": kredi,
        "yas": yas,
        "harcama_skoru": harcama,
        "meslek": meslek,
        "segment": segment,
        "sehir": sehir,
        "kredi_onay": hedef
    })

    # Eksik veri (Missing value) enjeksiyonu
    mask_gelir = np.random.rand(n) < 0.08
    df.loc[mask_gelir, "gelir"] = np.nan
    mask_meslek = np.random.rand(n) < 0.05
    df.loc[mask_meslek, "meslek"] = np.nan

    sayisal_kolonlar = ["gelir", "kredi_tutari", "yas", "harcama_skoru"]
    kategorik_kolonlar = ["meslek", "segment", "sehir"]

    print(f"[+] Veri Seti Üretildi: {df.shape[0]} Satır x {df.shape[1]} Kolon")
    print(f"    - Sayısal Kolonlar  : {sayisal_kolonlar}")
    print(f"    - Kategorik Kolonlar: {kategorik_kolonlar}")
    print(f"    - Eksik Değer Sayısı: Gelir ({df['gelir'].isna().sum()}), Meslek ({df['meslek'].isna().sum()})")

    # 2. Hedef Sızıntısı (Target Leakage) Denetimi
    print("\n[+] 1. Adım: Hedef Sızıntısı ve Şüpheli Korelasyon Taraması...")
    dedektor = TargetLeakageDedektoru(korelasyon_esigi=0.88)
    sizinti_raporu = dedektor.denetle(df, hedef_kolon="kredi_onay")
    print(f"    - Sızıntı Durumu: {sizinti_raporu['durum']}")
    print(f"    - Şüpheli Kolon : {sizinti_raporu['supheli_kolon_sayisi']} Adet")

    # 3. Nested Cross-Validation ile Güvenli Model Doğrulama
    print("\n[+] 2. Adım: 5-Katman Dış x 3-Katman İç Nested Cross-Validation Yürütülüyor...")
    X = df.drop(columns=["kredi_onay"])
    y = df["kredi_onay"]

    param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
    nested_motor = NestedCVMotoru(outer_splits=5, inner_splits=3, random_state=42)

    nested_sonuc = nested_motor.nested_cv_yurut(
        X=X,
        y=y,
        sayisal_kolonlar=sayisal_kolonlar,
        kategorik_kolonlar=kategorik_kolonlar,
        param_grid=param_grid
    )
    print(f"    - Dış Katman Skorları (ROC-AUC): {nested_sonuc['outer_skorlar']}")
    print(f"    - Güvenli Ortalama ROC-AUC     : %{nested_sonuc['ortalama_auc'] * 100:.2f} (±{nested_sonuc['std_auc']:.3f})")

    # 4. Sızıntılı Naive Ön İşleme ile Karşılaştırma
    print("\n[+] 3. Adım: Sızıntılı (Leaky) Naive Ön İşleme Simülasyonu...")
    leaky_sonuc = nested_motor.sizintili_karsilastirma_yurut(X, y, sayisal_kolonlar)
    print(f"    - Sızıntılı Ortalama ROC-AUC   : %{leaky_sonuc['leaky_ortalama_auc'] * 100:.2f} (±{leaky_sonuc['leaky_std_auc']:.3f})")
    print(f"    - Sızıntı Yanlılık Farkı       : +{leaky_sonuc['leaky_ortalama_auc'] - nested_sonuc['ortalama_auc']:.4f} AUC Şişmesi")

    # 5. Tam Veri Üzerinde Güvenli Pipeline Eğitimi ve Katsayı Analizi
    print("\n[+] 4. Adım: Tüm Veride Güvenli Pipeline Eğitimi...")
    son_pipeline = GuvenliPipelineUretici.guvenli_pipeline_olustur(
        sayisal_kolonlar=sayisal_kolonlar,
        kategorik_kolonlar=kategorik_kolonlar,
        model_turu="logistic",
        c_param=1.0,
        random_state=42
    )
    son_pipeline.fit(X, y)

    # Özellik İsimleri ve Katsayıları Çıkarma
    preprocessor = son_pipeline.named_steps["preprocessor"]
    cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(kategorik_kolonlar)
    tum_feature_names = list(sayisal_kolonlar) + list(cat_feature_names)
    katsayilar_arr = son_pipeline.named_steps["classifier"].coef_[0]

    katsayilar_dict = {f: float(round(w, 4)) for f, w in zip(tum_feature_names, katsayilar_arr)}

    # 6. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85)
    print(">>> 5. 6 PANELLİ GÜVENLİ PIPELINE VE SIZINTI DENETİM PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)
    panel_yolu = PipelineTehisGorsellestirici.panel_ciz(
        nested_sonuclari=nested_sonuc,
        leaky_sonuclari=leaky_sonuc,
        sizinti_raporu=sizinti_raporu,
        katsayilar=katsayilar_dict,
        hedef_path="day-47-sklearn-leakage-safe-ml-pipeline/ciktilar/leakage_guvenli_pipeline_paneli.png"
    )
    print(f"[+] 6 Panelli Pipeline Panosu Kaydedildi: {os.path.abspath(panel_yolu)}")
    print("=" * 85)
    print("DAY 47: GÜVENLİ PIPELINE VE NESTED CV PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
