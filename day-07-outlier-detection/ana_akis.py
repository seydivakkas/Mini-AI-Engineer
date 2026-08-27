"""Günün Ana Çalıştırma Akışı: İstatistiksel ve Makine Öğrenmesi Tabanlı Aykırı Değer Tespiti.

Bu betik; endüstriyel kamera sensör verileri üzerinde Z-Skoru, IQR,
İzolasyon Ormanı ve LOF algoritmalarını koşturarak tespitleri karşılaştırır,
mutabakat oylaması yapar ve 2x2 karşılaştırma grafiğini disk üzerine kaydeder.
"""

import sys
from pathlib import Path

# Proje kök dizinini modül yoluna ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
import pandas as pd
from src.istatistiksel_tespit import ZSkoruTespitEdici, IqrAykiriDegerTespitEdici
from src.makine_ogrenmesi_tespiti import IzolasyonOrmaniTespitEdici, LokalAykiriFaktorTespitEdici
from src.karsilastirma_ve_gorsellestirme import AykiriDegerKarsilastirici, AykiriDegerGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def main() -> None:
    # 1. Gerçekçi Çok Kümeli Endüstriyel Sensör Verisi Simülasyonu
    np.random.seed(42)

    # Küme A: Yüksek Yoğunluklu Hat (300 örnek)
    kume_a = np.random.normal(loc=[45.0, 15.0], scale=[3.0, 2.0], size=(300, 2))

    # Küme B: Düşük Yoğunluklu / Seyrek Hat (200 örnek)
    kume_b = np.random.normal(loc=[70.0, 35.0], scale=[6.0, 5.0], size=(200, 2))

    # Küresel Aykırılar (Global Outliers): Dağılımın fersah fersah dışındaki arızalar (15 örnek)
    kuresel_aykirilar = np.random.uniform(low=[100.0, 60.0], high=[130.0, 85.0], size=(15, 2))

    # Yerel Aykırılar (Local Outliers): Küme A'nın sınırında kalan ama yoğunluğu çok düşük olanlar (10 örnek)
    yerel_aykirilar = np.array([
        [56.0, 23.0], [58.0, 24.0], [35.0, 8.0], [36.0, 9.0], [55.0, 10.0],
        [57.0, 11.0], [40.0, 22.0], [38.0, 21.0], [59.0, 19.0], [60.0, 18.0]
    ])

    tum_veri = np.vstack([kume_a, kume_b, kuresel_aykirilar, yerel_aykirilar])
    toplam_nokta = len(tum_veri)

    baslik("AŞAMA 1: Sentetik Endüstriyel Sensör Verisi")
    print(f"Toplam Örnek Sayısı          : {toplam_nokta}")
    print(f"Normal Kümeler (A ve B)       : 500 nokta")
    print(f"Enjekte Edilen Küresel Aykırı : 15 nokta")
    print(f"Enjekte Edilen Yerel Aykırı   : 10 nokta")
    print(f"Öznitelikler                  : [Sensör Sıcaklığı (°C), Titreşim Şiddeti (mm/s)]")

    baslik("AŞAMA 2: 4 Temel Algoritmanın Çalıştırılması ve Karşılaştırılması")
    karsilastirici = AykiriDegerKarsilastirici(tum_veri)
    sonuclar = karsilastirici.tum_yontemleri_calistir(kirlilik_orani=0.05)

    print(f"{'Yöntem Adı':<26} | {'Aykırı Sayısı':<15} | {'Yüzde Oranı'}")
    print("-" * 74)
    for yontem_adi, maske in sonuclar.items():
        sayi = int(np.sum(maske))
        yuzde = (sayi / toplam_nokta) * 100
        print(f"{yontem_adi:<26} | {sayi:<15} | %{yuzde:.2f}")
    print("-" * 74)

    baslik("AŞAMA 3: Topluluk Mutabakatı (Ensemble Consensus) Analizi")
    mutabakat = karsilastirici.mutabakat_analizi(sonuclar)
    for aciklama, adet in mutabakat["oy_dagilimi"].items():
        yuzde = (adet / toplam_nokta) * 100
        print(f"  * {aciklama:<35}: {adet:>3} adet (%{yuzde:>5.2f})")

    baslik("AŞAMA 4: 2x2 Karşılaştırmalı Görselleştirme Çıktısı")
    cikti_yolu = proje_kok / "ciktilar" / "aykiri_deger_karsilastirma.png"
    kaydedilen = AykiriDegerGorsellestirici.karsilastirma_grafigi_ciz(
        tum_veri,
        sonuclar,
        cikti_yolu,
        x_etiketi="Sensör Sıcaklığı (°C)",
        y_etiketi="Titreşim Şiddeti (mm/s)"
    )
    print(f"[V] Karşılaştırma grafiği başarıyla kaydedildi: {kaydedilen.name}")
    print(f"[V] Tam Dosya Yolu: {kaydedilen}")
    print("\n[V] Day 7: İstatistiksel ve ML Tabanlı Aykırı Değer Tespiti tamamlandı.")


if __name__ == "__main__":
    main()
