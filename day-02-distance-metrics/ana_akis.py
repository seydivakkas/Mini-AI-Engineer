"""Günün Ana Çalıştırma Akışı: Vektörel ve Piksel Mesafe/Benzerlik Metrikleri Laboratuvarı.

Bu betik; Öklid, Manhattan, Chebyshev, Minkowski ve Kosinüs benzerlik metriklerini
karşılaştırmalı olarak çalıştırır, büyüklük yanlılığı (magnitude bias) deneyini gösterir,
piksel fark haritası çıkarır ve en yakın komşu (k-NN) katalog aramasını simüle eder.
"""

import sys
from pathlib import Path

# Proje kök dizinini modül arama yoluna ekler
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
from src.mesafe_olcer import MesafeOlcer
from src.gorsel_eslestirici import GorselBenzerlikEslestirici


def bolum_basligi(baslik: str) -> None:
    """Konsol çıktılarını yapılandırılmış bölümlere ayırır."""
    cizgi = "=" * 70
    print(f"\n{cizgi}\n>>> {baslik}\n{cizgi}")


def main() -> None:
    bolum_basligi("AŞAMA 1: Temel Vektörler Üzerinde Tüm Metriklerin Karşılaştırılması")
    # İki örnek görsel öznitelik (feature embedding) vektörü
    vektor_1 = np.array([0.25, 0.70, 0.10, 0.95, 0.40], dtype=np.float32)
    vektor_2 = np.array([0.30, 0.65, 0.15, 0.85, 0.50], dtype=np.float32)

    print(f"Vektör 1: {vektor_1}")
    print(f"Vektör 2: {vektor_2}")

    metrikler = MesafeOlcer.tum_metrikleri_hesapla(vektor_1, vektor_2)
    print("-" * 70)
    print(f"{'Metrik Adı':<25} | {'Değer':<12} | {'Ölçek Tipi':<15} | {'Yorum'}")
    print("-" * 70)
    for kod, sonuc in metrikler.items():
        yorum = "Sıfıra yakınsa benzer" if sonuc.olcek_tipi == "mesafe" else "1.0'a yakınsa benzer"
        print(f"{sonuc.metrik_adi:<25} | {sonuc.deger:<12.5f} | {sonuc.olcek_tipi:<15} | {yorum}")
    print("-" * 70)

    bolum_basligi("AŞAMA 2: Büyüklük Yanlılığı (Magnitude Bias) Deneyi: Öklid vs. Kosinüs")
    # İki vektör tamamen AYNI doğrultuda (yönde), ancak B vektörü A'nın 10 katı büyüklüğünde
    # (Örnek: Aynı fotoğrafın düşük ışıklı hali ile yüksek ışıklı hali)
    vektor_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vektor_b = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    oklid = MesafeOlcer.oklid_mesafesi(vektor_a, vektor_b)
    kosinus_bzn = MesafeOlcer.kosinus_benzerligi(vektor_a, vektor_b)

    print(f"Vektör A (Düşük Parlaklık) : {vektor_a}")
    print(f"Vektör B (Yüksek Parlaklık): {vektor_b}")
    print(f"[!] Öklid Mesafesi (L2)    : {oklid:.4f}  (Büyüklük farkından ötürü çok uzak görünüyor!)")
    print(f"[V] Kosinüs Benzerliği     : {kosinus_bzn:.4f}  (Yönler özdeş olduğu için KUSURSUZ eşleşme!)")
    print(">>> Çıkarım: Derin öğrenme embedding'lerinde l2-normalizasyonu yapılmadıkça")
    print("    kosinüs benzerliği parlaklık/ölçek değişimlerine karşı çok daha gürbüzdür.")

    bolum_basligi("AŞAMA 3: İki Görsel Arasında Piksel Düzeyinde Uzamsal Fark Haritası")
    # 64x64 boyutunda iki sentetik görsel üretelim
    taban_gorsel = np.zeros((64, 64, 3), dtype=np.uint8)
    taban_gorsel[:, :] = [100, 150, 200]

    degistirilmis_gorsel = taban_gorsel.copy()
    # Ortasına 20x20 boyutunda bir kırmızı kusur/anomali ekleyelim
    degistirilmis_gorsel[22:42, 22:42] = [255, 0, 0]

    fark_haritasi = MesafeOlcer.piksel_fark_haritasi(taban_gorsel, degistirilmis_gorsel, metrik="oklid")
    print(f"[+] Fark Haritası Boyutu   : {fark_haritasi.shape}")
    print(f"[+] Min Piksel Farkı       : {fark_haritasi.min():.2f}")
    print(f"[+] Max Piksel Farkı       : {fark_haritasi.max():.2f} (Kusurlu bölgedeki Öklid renk farkı)")
    print(f"[+] Kusurlu Piksel Adedi   : {np.sum(fark_haritasi > 0)} piksel (Beklenen: 20x20 = 400)")

    bolum_basligi("AŞAMA 4: Görsel Benzerlik Motoru ile Katalog Araması (Top-3 Retrieval)")
    eslestirici = GorselBenzerlikEslestirici(metrik="kosinus")

    # 5 adet referans desen vektörü kataloga ekleyelim
    np.random.seed(42)
    katalog = {
        "Hali_Klasik_Hereke": np.array([0.9, 0.1, 0.05, 0.8, 0.1], dtype=np.float32),
        "Hali_Modern_Geometrik": np.array([0.1, 0.8, 0.85, 0.2, 0.9], dtype=np.float32),
        "Hali_Minimalist_Iskandinav": np.array([0.15, 0.75, 0.80, 0.25, 0.85], dtype=np.float32),
        "Hali_Vintage_Usak": np.array([0.85, 0.15, 0.10, 0.75, 0.2], dtype=np.float32),
        "Hali_Ipek_Kayseri": np.array([0.75, 0.25, 0.20, 0.85, 0.15], dtype=np.float32),
    }

    for kimlik, ozellik in katalog.items():
        eslestirici.katalog_ekle(kimlik, ozellik)

    # Müşterinin arattığı sorgu görseli (Vintage Uşak desenine yakın)
    sorgu = np.array([0.88, 0.12, 0.08, 0.78, 0.18], dtype=np.float32)
    print("Sorgu Vektörü kataloga soruluyor...\n")
    en_iyiler = eslestirici.en_yakin_k_bul(sorgu, k=3)

    for sonuc in en_iyiler:
        print(f"  #{sonuc.sira} Eşleşme: {sonuc.oge_kimligi:<26} -> Skor: {sonuc.skor:.4f} ({sonuc.metrik})")

    print("\n[V] Day 2: Vektörel ve Piksel Mesafe Metrikleri başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
