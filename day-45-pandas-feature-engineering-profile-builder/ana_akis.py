"""
Day 45: Özellik Mühendisliği, Encoding, Ölçeklendirme ve Feature Store Profil Oluşturucu Ana Yürütme Betiği.
"""

import os
import numpy as np
import pandas as pd
from src.kodlayicilar import KategorikKodlayici
from src.olcekleyiciler import SayisalOlcekleyici
from src.ozellik_profili import FeatureStoreProfilci
from src.gorsellestirici import OzellikMuhendisligiGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 45: ÖZELLİK MÜHENDİSLİĞİ, ENCODING, ÖLÇEKLENDİRME VE FEATURE STORE PROFİLİ")
    print("=" * 85)

    # 1. Ham Tabüler Finansal Risk Veri Setinin Simülasyonu
    np.random.seed(42)
    n = 500

    ham_df = pd.DataFrame({
        "musteri_id": np.arange(1000, 1000 + n),
        "gelir": np.random.lognormal(mean=9.8, sigma=0.75, size=n),  # Sağa çarpık gelir
        "kredi_tutari": np.random.uniform(5000.0, 120000.0, size=n),
        "yas": np.random.randint(21, 68, size=n),
        "meslek_grubu": np.random.choice(["MUHENDIS", "DOKTOR", "MEMUR", "SERBEST", "OGRENCI", "EMEKLİ"], size=n),
        "sehir": np.random.choice(["ISTANBUL", "ANKARA", "IZMIR", "BURSA", "ANTALYA"], size=n),
        "risk_skoru": np.random.beta(a=2.0, b=5.0, size=n)  # 0 ile 1 arası hedef değişken
    })

    print(f"[+] Ham Veri Seti Oluşturuldu: {ham_df.shape[0]} Satır x {ham_df.shape[1]} Kolon")

    # 2. Kategorik Özellik Kodlama (Target, Frequency, OHE)
    print("\n[+] 1. Adım: Kategorik Özellik Kodlama (Target, Frequency & OHE)...")
    kodlayici = KategorikKodlayici(smoothing_weight=15.0)

    # Smoothed Target Encoding
    kodlayici.fit_target_encoding(ham_df, kolonlar=["meslek_grubu", "sehir"], hedef_kolon="risk_skoru")
    df_islenmis = kodlayici.transform_target_encoding(ham_df, kolonlar=["meslek_grubu", "sehir"])

    # Frequency Encoding
    kodlayici.fit_frequency_encoding(df_islenmis, kolonlar=["sehir"])
    df_islenmis = kodlayici.transform_frequency_encoding(df_islenmis, kolonlar=["sehir"])

    # One-Hot Encoding
    df_islenmis = kodlayici.fit_transform_one_hot(df_islenmis, kolonlar=["meslek_grubu"], drop_first=True)

    # 3. Sayısal Dönüşüm ve Ölçeklendirme (Log1p, StandardScaler, RobustScaler, Oranlar)
    print("[+] 2. Adım: Sayısal Ölçeklendirme ve Etkileşim Terimleri Üretimi...")
    olcekleyici = SayisalOlcekleyici()

    # Log1p Dönüşümü
    df_islenmis = olcekleyici.log1p_donusumu(df_islenmis, kolonlar=["gelir"])

    # Standart ve Robust Ölçeklendirme
    olcekleyici.fit_standard_scaler(df_islenmis, kolonlar=["yas"])
    df_islenmis = olcekleyici.transform_standard_scaler(df_islenmis, kolonlar=["yas"])

    olcekleyici.fit_robust_scaler(df_islenmis, kolonlar=["kredi_tutari"])
    df_islenmis = olcekleyici.transform_robust_scaler(df_islenmis, kolonlar=["kredi_tutari"])

    # Domain Etkileşim Oranı: Borç / Gelir Oranı
    df_islenmis = olcekleyici.etkilesim_ve_oran_uret(
        df_islenmis,
        pay_kolon="kredi_tutari",
        payda_kolon="gelir",
        yeni_ad="borc_gelir_orani"
    )

    print(f"[+] Özellik Mühendisliği Tamamlandı: {len(ham_df.columns)} Ham Kolon -> {len(df_islenmis.columns)} Türetilmiş Özellik (+{len(df_islenmis.columns) - len(ham_df.columns)})")

    # 4. Feature Store Profilleme ve Metadata Kataloğu
    print("\n" + "=" * 85)
    print(">>> 3. FEATURE STORE METADATA VE PROFİLLEME ÇIKTISI:")
    print("=" * 85)
    profil_raporu = FeatureStoreProfilci.profil_cikar(df_islenmis, hedef_kolon="risk_skoru")

    print(f"    - Toplam Özellik Sayısı : {profil_raporu['toplam_oznitelik_sayisi']} Adet")
    print(f"    - Sayısal Özellikler   : {profil_raporu['sayisal_oznitelik_sayisi']} Adet")
    print(f"    - Kategorik Özellikler : {profil_raporu['kategorik_oznitelik_sayisi']} Adet")

    print("\n    [*] Seçilmiş Örnek Özellik Metadata Kayıtları:")
    for col in ["gelir_log1p", "meslek_grubu_target_enc", "borc_gelir_orani", "yas_std_scaled"]:
        meta = profil_raporu["oznitelikler"][col]
        print(f"      - {col:<24} | Min: {meta.get('min', 0):6.2f} | Max: {meta.get('max', 0):6.2f} | Skew: {meta.get('carpiklik_skew', 0):5.2f} | Corr: {meta.get('hedef_korelasyonu', 0):6.3f}")

    feast_sema = FeatureStoreProfilci.feast_sema_ihrac_et(profil_raporu, feature_view_adi="musteri_risk_view")
    print(f"\n[+] Feast-Uyumlu Feature View Sözlüğü Üretildi: {len(feast_sema['features'])} Özellik Kayıtlı")

    # -------------------------------------------------------------
    # 5. 6 Panelli Teşhis Panosunun Çizilmesi
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> 5. KONSOLİDE 6 PANELLİ ÖZELLİK MÜHENDİSLİĞİ TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_resmi = OzellikMuhendisligiGorsellestirici.panel_ciz(
        ham_df=ham_df,
        islenmis_df=df_islenmis,
        profil_raporu=profil_raporu,
        hedef_kolon="risk_skoru",
        hedef_path="day-45-pandas-feature-engineering-profile-builder/ciktilar/ozellik_muhendisligi_paneli.png"
    )
    print(f"[+] 6 Panelli Özellik Mühendisliği Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 85)
    print("DAY 45: ÖZELLİK MÜHENDİSLİĞİ VE FEATURE STORE PROFİLİ PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
