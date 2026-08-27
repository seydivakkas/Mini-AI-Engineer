"""Günün Ana Çalıştırma Akışı: Mahalanobis vs. Öklid Karşılaştırma Laboratuvarı.

Bu betik; yüksek korelasyonlu çok değişkenli bir veri kümesi oluşturur,
Öklid mesafesinin aldandığı sınır senaryoları kanıtlar ve Ki-Kare dağılımı
destekli istatistiksel anomali tespitini baştan sona icra eder.
"""

import sys
from pathlib import Path

# Proje kök dizinini modül arama yoluna ekler
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
from src.kovaryans_ve_mesafe import KovaryansAnalizoru, MahalanobisMesafeOlcer
from src.anomali_tespit_edici import MahalanobisAnomaliDedektoru


def baslik_yazdir(baslik: str) -> None:
    """Bölüm başlığını konsola biçimlendirerek yazar."""
    cizgi = "=" * 72
    print(f"\n{cizgi}\n>>> {baslik}\n{cizgi}")


def main() -> None:
    baslik_yazdir("AŞAMA 1: Yüksek Korelasyonlu Çok Değişkenli Veri Dağılımı Oluşturma")
    # Endüstriyel Kalite Kontrol Örneği:
    # Değişken 1: Kumaş İplik Yoğunluğu (Ort: 100)
    # Değişken 2: Kumaş Ağırlığı (Ort: 50)
    # İplik yoğunluğu arttıkça ağırlık da doğal olarak artar (Güçlü Pozitif Korelasyon r ≈ 0.9)
    np.random.seed(42)
    ornek_sayisi = 1000
    ortalama_hedef = np.array([100.0, 50.0])
    hedef_kovaryans = np.array([
        [16.0, 7.2],   # Var(X1) = 16, Cov(X1, X2) = 7.2
        [7.2, 4.0]     # Cov(X1, X2) = 7.2, Var(X2) = 4.0 -> r = 7.2 / (4 * 2) = 0.90!
    ])

    sentetik_veri = np.random.multivariate_normal(ortalama_hedef, hedef_kovaryans, size=ornek_sayisi)

    kovaryans_matrisi = KovaryansAnalizoru.kovaryans_matrisi_hesapla(sentetik_veri)
    korelasyon_matrisi = KovaryansAnalizoru.korelasyon_matrisi_hesapla(kovaryans_matrisi)

    print(f"[+] Örneklem Boyutu           : {sentetik_veri.shape[0]} satır, {sentetik_veri.shape[1]} sütun")
    print(f"[+] Hesaplanan Ortalama Vektör : {np.mean(sentetik_veri, axis=0).round(2)}")
    print(f"[+] Kovaryans Matrisi (Sigma) :\n{kovaryans_matrisi.round(3)}")
    print(f"[+] Korelasyon Katsayısı (r)  : {korelasyon_matrisi[0, 1]:.3f} (Yüksek Pozitif Korelasyon)")

    baslik_yazdir("AŞAMA 2: Öklid vs. Mahalanobis Paradoksu Deneyi")
    olcer = MahalanobisMesafeOlcer(sentetik_veri)

    # İki kritik test noktası tanımlıyoruz:
    # Nokta A: Dağılımın korelasyon ekseni doğrultusunda uzakta (İplik ve ağırlık birlikte artmış)
    # Nokta B: Korelasyon eksenine dik konumda (İplik çok yüksek ama ağırlık tuhaf şekilde düşük kalmış!)
    test_noktalari = {
        "Nokta_A_Dogal_Egim": np.array([108.0, 54.0]),    # Eksen boyunca hareket
        "Nokta_B_Anormal_Fark": np.array([100.0, 58.94]), # Merkeze Öklidçe eşit ama korelasyona aykırı!
    }

    rapor = olcer.kiyaslama_raporu_olustur(test_noktalari)
    print(f"{'Test Noktası':<24} | {'Öklid (L2)':<12} | {'Mahalanobis':<14} | {'Öklid Sıra':<11} | {'Maha Sıra'}")
    print("-" * 72)
    for isim, sonuc in rapor.items():
        print(
            f"{isim:<24} | {sonuc.oklid_mesafesi:<12.4f} | {sonuc.mahalanobis_mesafesi:<14.4f} | "
            f"#{sonuc.oklid_sirasi:<10} | #{sonuc.mahalanobis_sirasi}"
        )
    print("-" * 72)
    print(">>> DİKKAT: Nokta A ve Nokta B'nin merkeze Öklid mesafeleri neredeyse AYNI (~8.94)!")
    print("    Ancak Mahalanobis Nokta B'yi korelasyonu ihlal ettiği için çok daha uzak (anormal) buldu!")

    baslik_yazdir("AŞAMA 3: Ki-Kare İstatistiği ile Otomatik Anomali Tespiti")
    # %99 güven seviyesinde (alpha = 0.01) anomali dedektörü eğitelim
    dedektor = MahalanobisAnomaliDedektoru(anlamlilik_duzeyi=0.01)
    dedektor.egit(sentetik_veri)

    print(f"[+] Serbestlik Derecesi (D)     : {dedektor.serbestlik_derecesi}")
    print(f"[+] Belirlenen Eşik Mesafesi (tau): {dedektor.esik_mesafe:.4f} (Chi2 %99 kritik değeri)")

    ornekler = np.array([
        [101.0, 50.5],   # Kusursuz normal ürün
        [108.0, 54.0],   # Normal sınırda ürün (Nokta A)
        [100.0, 58.94],  # Kusurlu ürün (Nokta B - fiziksel üretim hatası)
        [120.0, 70.0],   # Çok aşırı aykırı ürün
    ])

    etiketler = ["Normal Ürün", "Sınırda Normal", "Korelasyon Kusuru", "Aşırı Aykırı"]
    tahminler = dedektor.tahmin_et(ornekler)

    print("-" * 72)
    print(f"{'Örnek Tipi':<20} | {'Maha Mesafesi':<15} | {'Eşik':<8} | {'p-Değeri':<10} | {'Karar'}")
    print("-" * 72)
    for etiket, t in zip(etiketler, tahminler):
        karar_metni = "[ANOMALİ]" if t.anomali_mi else "[NORMAL]"
        print(
            f"{etiket:<20} | {t.mahalanobis_mesafesi:<15.4f} | {t.esik_degeri:<8.4f} | "
            f"{t.p_degeri:<10.6f} | {karar_metni}"
        )
    print("-" * 72)
    print("\n[V] Day 3: Mahalanobis vs. Öklid Analizi başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
