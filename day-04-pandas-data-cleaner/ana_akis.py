"""Günün Ana Çalıştırma Akışı: Pandas Tabüler Veri Temizleme ve Ön İşleme Laboratuvarı.

Bu betik; kirli endüstriyel görsel metaverisi üretir, veri sızıntısını (Data Leakage)
engelleyerek fit-transform mimarisiyle temizler, aykırı değerleri IQR ile sınırlar
ve bellek tüketimini optimize ederek ayrıntılı bir temizlik raporu sunar.
"""

import sys
from pathlib import Path

# Proje kök dizinini modül arama yoluna ekler
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import pandas as pd
from src.sentetik_veri_ureticisi import kirli_veri_kumesi_uret
from src.veri_temizleyici import TabulerVeriTemizleyici


def bolum_yazdir(baslik: str) -> None:
    """Konsol başlıklarını biçimlendirir."""
    cizgi = "=" * 72
    print(f"\n{cizgi}\n>>> {baslik}\n{cizgi}")


def main() -> None:
    bolum_yazdir("AŞAMA 1: Kirli ve Kusurlu Sentetik Tablonun İncelenmesi")
    ham_veri = kirli_veri_kumesi_uret(satir_sayisi=1000, rastgele_tohum=42)

    print(f"[+] Ham Veri Boyutu           : {ham_veri.shape[0]} satır, {ham_veri.shape[1]} sütun")
    print(f"[+] Ham Bellek Tüketimi       : {ham_veri.memory_usage(deep=True).sum() / 1024.0:.2f} KB")
    print(f"[+] Mükerrer (Duplicate) Satır: {ham_veri.duplicated().sum()} adet")
    print("\n[+] Sütun Bazlı Eksik Veri (NaN) Sayıları:")
    for sutun, eksik in ham_veri.isna().sum().items():
        if eksik > 0:
            print(f"    - {sutun:<20}: {eksik:>4} eksik (%{(eksik / len(ham_veri)) * 100:.1f})")

    print("\n[+] Ham Tablodan Örnek Aykırı Değerler:")
    print(f"    - Min Genişlik            : {ham_veri['genislik'].min():.1f} (Negatif hata!)")
    print(f"    - Max Genişlik            : {ham_veri['genislik'].max():.1f} (Uç anomali!)")
    print(f"    - Max Parlaklık           : {ham_veri['ortalama_parlaklik'].max():.1f} (255 sınırını aşmış!)")

    bolum_yazdir("AŞAMA 2: Veri Sızıntısız Temizleme (Fit-Transform Ayrımı)")
    # Veri setini Eğitim (%80) ve Test (%20) olarak bölelim
    egitim_orani = 0.80
    kesme_noktasi = int(len(ham_veri) * egitim_orani)
    egitim_verisi = ham_veri.iloc[:kesme_noktasi].copy()
    test_verisi = ham_veri.iloc[kesme_noktasi:].copy()

    temizleyici = TabulerVeriTemizleyici()

    # Parametreleri (medyan, mod, IQR) YALNIZCA eğitim setinden öğreniyoruz!
    temizleyici.fit(egitim_verisi, sayisal_strateji="medyan", kategorik_strateji="mod", iqr_carpani=1.5)
    print("[V] Eğitim seti üzerinde istatistiksel parametreler öğrenildi (fit).")

    # Öğrenilen parametreleri hem eğitime hem de teste bağımsız uyguluyoruz
    temiz_egitim = temizleyici.transform(egitim_verisi)
    temiz_test = temizleyici.transform(test_verisi)

    print(f"[V] Eğitim Verisi Temizlendi: {temiz_egitim.shape}")
    print(f"[V] Test Verisi Temizlendi  : {temiz_test.shape}")

    bolum_yazdir("AŞAMA 3: Temizlik ve Optimizasyon Raporu (Eğitim Kümesi)")
    rapor = temizleyici.rapor_olustur(egitim_verisi, temiz_egitim)

    print(f"[+] Başlangıç Satır Sayısı    : {rapor.baslangic_satir_sayisi}")
    print(f"[+] Bitiş Satır Sayısı        : {rapor.bitis_satir_sayisi}")
    print(f"[+] Silinen Mükerrer Satırlar : {rapor.elenen_yineleme_sayisi} adet")

    print("\n[+] Tamamlanan Eksik Değerler:")
    for sutun, adet in rapor.tamamlanan_eksik_degerler.items():
        print(f"    - {sutun:<20}: {adet} adet dolduruldu")

    print("\n[+] IQR ile Sınırlandırılan Aykırı Değerler:")
    for sutun, adet in rapor.budanan_aykiri_degerler.items():
        print(f"    - {sutun:<20}: {adet} adet sınırlandı (clipping)")

    print("\n[+] Bellek Tasarrufu ve Tip Dönüşümü:")
    print(f"    - Başlangıç Bellek        : {rapor.baslangic_bellek_kb:.2f} KB")
    print(f"    - Bitiş Bellek            : {rapor.bitis_bellek_kb:.2f} KB")
    print(f"    - Net Bellek Kazancı      : %{rapor.bellek_kazanc_yuzdesi:.2f} TASARRUF!")

    bolum_yazdir("AŞAMA 4: Temizlenmiş Verinin Sağlamlık Kontrolü")
    print(f"[+] Kalan Eksik Veri Toplamı  : {temiz_egitim.isna().sum().sum()} (Hedef: 0)")
    print(f"[+] Kalan Mükerrer Satır Sayısı: {temiz_egitim.duplicated().sum()} (Hedef: 0)")
    print(f"[+] Yeni Genişlik Aralığı     : [{temiz_egitim['genislik'].min():.1f}, {temiz_egitim['genislik'].max():.1f}]")
    print(f"[+] Yeni Parlaklık Aralığı    : [{temiz_egitim['ortalama_parlaklik'].min():.1f}, {temiz_egitim['ortalama_parlaklik'].max():.1f}]")
    print("\n[+] Optimize Edilmiş Veri Tipleri (Dtypes):")
    for sutun, tip in temiz_egitim.dtypes.items():
        print(f"    - {sutun:<20}: {tip}")

    print("\n[V] Day 4: Pandas Tabüler Veri Temizleme başarıyla icra edildi.")


if __name__ == "__main__":
    main()
