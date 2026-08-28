"""
Day 44: Pandas ile Üretim Seviyesi Şema Doğrulama ve Otomatik Veri Kalitesi Temizliği Ana Yürütme Betiği.
"""

import os
import numpy as np
import pandas as pd
from src.sema import KolonKurali, TabloSemasi
from src.dogrulayici import SemaDogrulayici
from src.temizleyici import OtomatikVeriTemizleyici
from src.gorsellestirici import VeriKaliteGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 44: PANDAS İLE ÜRETİM SEVİYESİ ŞEMA DOĞRULAMA VE OTOMATİK VERİ TEMİZLİĞİ")
    print("=" * 85)

    # 1. Üretim Tablo Şemasının Deklaratif Olarak Tanımlanması
    sema = TabloSemasi(
        tablo_adi="MusteriRiskAnalizTablosu",
        kolon_kurallari=[
            KolonKurali(ad="musteri_id", tip=int, zorunlu=True, benzersiz=True, izin_verilen_null_orani=0.0),
            KolonKurali(ad="yas", tip=int, zorunlu=True, min_deger=18.0, max_deger=90.0, varsayilan_doldurma="median"),
            KolonKurali(ad="aylik_harcama", tip=float, zorunlu=True, min_deger=0.0, max_deger=50000.0, varsayilan_doldurma="mean"),
            KolonKurali(ad="segment", tip=str, zorunlu=True, kategoriler=["BRONZ", "GUMUS", "ALTIN", "VIP"], varsayilan_doldurma="mode"),
            KolonKurali(ad="eposta", tip=str, zorunlu=False, regex_kalibi=r"^[\w\.-]+@[\w\.-]+\.\w+$", varsayilan_doldurma="destek@sirket.com")
        ],
        beklenmeyen_kolon_engeli=False,
        izin_verilen_cift_satir_orani=0.0
    )

    dogrulayici = SemaDogrulayici(sema)
    temizleyici = OtomatikVeriTemizleyici(sema)

    print("[+] Deklaratif Şema Sözleşmesi Devreye Alındı:")
    print("    - Tablo: MusteriRiskAnalizTablosu")
    print("    - Kurallar: [musteri_id: unique, yas: [18, 90], harcama: [0, 50000], segment: Categorical]")

    # -------------------------------------------------------------
    # Senaryo 1: Kusursuz Üretim Tablosu (Golden Table)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 1: KUSURSUZ ÜRETİM TABLOSU (GOLDEN TABLE)")
    print("=" * 85)
    np.random.seed(42)
    n1 = 300
    kusursuz_df = pd.DataFrame({
        "musteri_id": np.arange(1000, 1000 + n1),
        "yas": np.random.randint(20, 65, size=n1),
        "aylik_harcama": np.random.uniform(500.0, 15000.0, size=n1),
        "segment": np.random.choice(["BRONZ", "GUMUS", "ALTIN", "VIP"], size=n1),
        "eposta": [f"user_{i}@ornek.com" for i in range(n1)]
    })

    rapor1 = dogrulayici.dogrula(kusursuz_df)
    print(f"    - Karar          : {rapor1['karar']}")
    print(f"    - Kalite Skoru   : %{rapor1['kalite_skoru']:.1f} / 100")
    print(f"    - İhlal Sayısı   : {rapor1['toplam_ihlal_sayisi']} Adet")
    print(f"    - Denetim Süresi : {rapor1['denetim_suresi_ms']:.2f} ms")

    # -------------------------------------------------------------
    # Senaryo 2: Düzeltilebilir Kirli Tablo (Outliers, Nulls, Duplicates, Malformed Strings)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 2: DÜZELTİLEBİLİR KİRLİ TABLO (OTOMATİK TEMİZLEME GEREKLİ)")
    print("=" * 85)
    kirli_df = kusursuz_df.copy()
    
    # Hatalar Enjekte Ediliyor
    kirli_df.loc[5:10, "yas"] = -8           # Sınır dışı negatif yaş
    kirli_df.loc[20:25, "yas"] = 150         # Aşırı büyük yaş
    kirli_df.loc[30:40, "aylik_harcama"] = np.nan # Eksik harcama
    kirli_df.loc[50:55, "segment"] = "GECERSIZ_KOD" # Geçersiz kategori
    kirli_df.loc[60:65, "eposta"] = "hatali_mail_adresi" # Regex format ihlali
    
    # 15 satır mükerrer ekle
    kirli_df = pd.concat([kirli_df, kirli_df.iloc[:15]], ignore_index=True)

    rapor2 = dogrulayici.dogrula(kirli_df)
    print(f"    - Karar          : {rapor2['karar']}")
    print(f"    - Kalite Skoru   : %{rapor2['kalite_skoru']:.1f} / 100")
    print(f"    - İhlal Sayısı   : {rapor2['toplam_ihlal_sayisi']} Adet")
    for ih in rapor2["ihlaller"][:4]:
        print(f"      [*] {ih['kod']} ({ih['kolon']}): {ih['mesaj']}")

    temiz_df, temiz_rapor = temizleyici.temizle_ve_iyilestir(kirli_df)
    print("\n    [+] Otomatik Veri Temizleme & İmpütasyon Sonucu:")
    print(f"      - Başlangıç Satır: {temiz_rapor['baslangic_satir_sayisi']} -> Temiz Satır: {temiz_rapor['temizlenmis_satir_sayisi']}")
    print(f"      - Yapılan İşlem  : {temiz_rapor['yapilan_islemler'][:3]}")

    # Temizlenmiş veriyi yeniden doğrula
    rapor2_sonrasi = dogrulayici.dogrula(temiz_df)
    print(f"      - Temizleme Sonrası Kalite Skoru: %{rapor2_sonrasi['kalite_skoru']:.1f} / 100 ({rapor2_sonrasi['karar']})")

    # -------------------------------------------------------------
    # Senaryo 3: Kritik Eksik Kolon Hatası
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 3: KRİTİK ŞEMA HATASI (ZORUNLU KOLON EKSİK)")
    print("=" * 85)
    bozuk_df = kusursuz_df.drop(columns=["musteri_id"])
    rapor3 = dogrulayici.dogrula(bozuk_df)
    print(f"    - Karar          : {rapor3['karar']}")
    print(f"    - Durum          : {rapor3['durum']}")
    print(f"    - Kritik Hata    : {rapor3['ihlaller'][0]['mesaj']}")

    # -------------------------------------------------------------
    # 5. 6 Panelli Teşhis Panosunun Üretilmesi
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> 5. KONSOLİDE 6 PANELLİ VERİ KALİTE TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_resmi = VeriKaliteGorsellestirici.panel_ciz(
        ham_df=kirli_df,
        temiz_df=temiz_df,
        denetim_raporu=rapor2,
        temizleme_raporu=temiz_rapor,
        hedef_path="day-44-pandas-data-quality-cleaner/ciktilar/veri_kalite_paneli.png"
    )
    print(f"[+] 6 Panelli Veri Kalitesi Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 85)
    print("DAY 44: PANDAS İLE VERİ KALİTESİ VE ŞEMA DOĞRULAMA PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
