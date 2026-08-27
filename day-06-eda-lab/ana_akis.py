"""Günün Ana Çalıştırma Akışı: Kapsamlı Keşifçi Veri Analizi Laboratuvarı.

Bu betik; endüstriyel görsel öznitelik tablosu üzerinde Pearson/Spearman korelasyonu,
çoklu doğrusallık (VIF) tespiti, hedef değişken ilişkileri ve grafiksel görselleştirmeleri
baştan sona icra ederek disk üzerine PNG ve konsol raporları sunar.
"""

import sys
from pathlib import Path

# Proje kök dizinini modül arama yoluna ekler
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
import pandas as pd
from src.kesifci_analizor import KesifciVeriAnalizoru
from src.grafik_ureteci import EdaGrafikUreteci


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def main() -> None:
    # 1. Gerçekçi Endüstriyel Dokuma & Görsel Veri Seti Oluşturma
    np.random.seed(42)
    ornek_sayisi = 800

    iplik_sikligi = np.random.normal(loc=120, scale=15, size=ornek_sayisi)
    # Ağırlık iplik sıklığı ile yüksek korelasyonludur (r ≈ 0.85)
    hali_agirligi = 15.0 + (0.45 * iplik_sikligi) + np.random.normal(0, 3, size=ornek_sayisi)
    # Düğüm sayısı formülle hesaplandığı için aşırı çoklu doğrusallık (Multicollinearity) yaratır!
    dugum_sayisi = (iplik_sikligi * 12.5) + np.random.normal(0, 5, size=ornek_sayisi)
    dokuma_hizi = np.random.uniform(5.0, 25.0, size=ornek_sayisi)
    parlaklik = np.random.normal(140, 20, size=ornek_sayisi)

    # Hedef Değişken: Hata Alanı (Dokuma hızı arttıkça ve sıklık düştükçe hata artar)
    hata_alani = np.maximum(
        0.0,
        (0.8 * dokuma_hizi) - (0.1 * iplik_sikligi) + np.random.normal(5, 4, size=ornek_sayisi)
    )

    kumas_tipleri = np.random.choice(["Akrilik", "Yun", "Ipek"], size=ornek_sayisi)

    veri = pd.DataFrame({
        "iplik_sikligi": iplik_sikligi,
        "hali_agirligi": hali_agirligi,
        "dugum_sayisi": dugum_sayisi,
        "dokuma_hizi": dokuma_hizi,
        "ortalama_parlaklik": parlaklik,
        "kusurlu_alan_mm2": hata_alani,
        "kumas_tipi": kumas_tipleri
    })

    baslik("AŞAMA 1: Korelasyon Matrisi ve Kritik Çiftlerin Tespiti")
    analizor = KesifciVeriAnalizoru(veri)
    kor_rapor = analizor.korelasyon_analizi(esik_degeri=0.70)

    print("Pearson Korelasyon Matrisi:")
    print(kor_rapor.pearson_matrisi)

    print("\n[!] Eşik Değerini Aşan Yüksek Korelasyonlu Çiftler (|r| >= 0.70):")
    for s1, s2, r_val, not_bilgisi in kor_rapor.yuksek_korelasyonlu_ciftler:
        print(f"    * {s1:<15} <---> {s2:<15} : r = {r_val:>6.2f} [{not_bilgisi}]")

    baslik("AŞAMA 2: Çoklu Doğrusallık (Multicollinearity) ve VIF Analizi")
    vif_sonuclari = analizor.vif_analizi()
    print(f"{'Öznitelik Adı':<22} | {'VIF Skoru':<12} | {'Risk Değerlendirmesi'}")
    print("-" * 74)
    for v in vif_sonuclari:
        print(f"{v.sututn_adi if hasattr(v, 'sututn_adi') else v.sutun_adi:<22} | {v.vif_degeri:<12.2f} | {v.durum}")
    print("-" * 74)
    print(">>> Kural: VIF > 10 olan sütunlar (ör. dugum_sayisi veya iplik_sikligi) model")
    print("    eğitilmeden önce elenmelidir; aksi halde model katsayıları kararsızlaşır.")

    baslik("AŞAMA 3: Hedef Değişken (Kusurlu Alan) İlişki Analizi")
    hedef_rapor = analizor.hedef_iliskisi_analizi(hedef_sutun="kusurlu_alan_mm2")
    print("Sayısal Değişkenlerin Kusur Alanı ile Korelasyonu:")
    for sutun, katsayi in hedef_rapor.sayisal_korelasyonlar.items():
        print(f"    * {sutun:<20}: {katsayi:>6.3f}")

    print("\nKumaş Tipine Göre Ortalama Kusurlu Alan Dağılımı:")
    for kat_sutun, oranlar in hedef_rapor.kategorik_dagilimlar.items():
        for kat, ort in oranlar.items():
            print(f"    - {kat_sutun} [{kat:<8}]: {ort:.2f} mm2")

    baslik("AŞAMA 4: Grafiksel Görselleştirmelerin Diske Kaydedilmesi")
    cikti_klasoru = proje_kok / "ciktilar"
    
    p1 = EdaGrafikUreteci.korelasyon_isi_haritasi(
        kor_rapor.pearson_matrisi,
        cikti_klasoru / "korelasyon_isi_haritasi.png"
    )
    p2 = EdaGrafikUreteci.dagilim_histogramlari(
        veri,
        ["iplik_sikligi", "hali_agirligi", "dokuma_hizi", "kusurlu_alan_mm2"],
        cikti_klasoru / "ozellik_dagilimlari.png"
    )
    p3 = EdaGrafikUreteci.sacilim_grafigi(
        veri,
        x_adi="dokuma_hizi",
        y_adi="kusurlu_alan_mm2",
        renk_sutunu="kumas_tipi",
        dosya_yolu=cikti_klasoru / "dokuma_hizi_vs_kusur_sacilim.png"
    )

    print(f"[V] 1. Isı Haritası Kaydedildi : {p1.name}")
    print(f"[V] 2. Histogramlar Kaydedildi : {p2.name}")
    print(f"[V] 3. Saçılım Grafiği Kaydedildi: {p3.name}")
    print("\n[V] Day 6: Kapsamlı Keşifçi Veri Analizi Laboratuvarı başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
