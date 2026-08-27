"""Günün Ana Çalıştırma Akışı: NumPy Görüntü Analizörü ve Piksel İstatistikleri.

Bu betik; sentetik görsel üretimini, bellek/adım (stride) denetimini, kanal ayrıştırma,
gri seviye dönüşümü, detaylı istatistik hesaplama ve üretim seviyesi normalizasyon
işlemlerini baştan sona çalıştırır ve konsola görselleştirilmiş bir rapor sunar.
"""

import sys
from pathlib import Path

# Proje kök dizinini modül arama yoluna ekler
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
from src.goruntu_analizoru import NumPyGoruntuAnalizoru
from src.yardimcilar import sentetik_goruntu_uret, bellek_ve_stride_raporla


def rapor_basligi_yazdir(baslik: str) -> None:
    """Konsola standartlaştırılmış bölüm başlığı basar."""
    cizgi = "=" * 65
    print(f"\n{cizgi}\n>>> {baslik}\n{cizgi}")


def main() -> None:
    rapor_basligi_yazdir("AŞAMA 1: Sentetik Test Görseli Üretimi ve Matris Boyutları")
    yukseklik, genislik = 128, 128
    sentetik_gorsel = sentetik_goruntu_uret(
        yukseklik=yukseklik,
        genislik=genislik,
        desen_tipi="renkli_bloklar"
    )
    print(f"[+] Üretilen Görsel Şekli (Shape)    : {sentetik_gorsel.shape} -> (Yükseklik, Genişlik, Kanal)")
    print(f"[+] Veri Tipi (Dtype)                : {sentetik_gorsel.dtype}")
    print(f"[+] Toplam Eleman Sayısı             : {sentetik_gorsel.size:,} değer")

    rapor_basligi_yazdir("AŞAMA 2: Düşük Seviyeli Bellek Yerleşimi ve Strides (Adımlar)")
    bellek_bilgisi = bellek_ve_stride_raporla(sentetik_gorsel)
    for anahtar, deger in bellek_bilgisi.items():
        print(f"  * {anahtar:<25}: {deger}")

    rapor_basligi_yazdir("AŞAMA 3: Analizör Başlatma ve Renk Kanalları Ayrıştırma")
    analizor = NumPyGoruntuAnalizoru(sentetik_gorsel)
    kanallar = analizor.kanal_ayristir()
    for kanal_adi, kanal_matrisi in kanallar.items():
        print(
            f"  * {kanal_adi:<10} Kanalı: Boyut={kanal_matrisi.shape}, "
            f"Min={kanal_matrisi.min():>3}, Max={kanal_matrisi.max():>3}, "
            f"Ortalama={kanal_matrisi.mean():>6.2f}"
        )

    rapor_basligi_yazdir("AŞAMA 4: Ağırlıklı Gri Ton Dönüşümü (ITU-R BT.601)")
    gri_matris = analizor.gri_tona_donustur()
    print(f"[+] Gri Matris Boyutu                : {gri_matris.shape}")
    print(f"[+] Gri Piksel Aralığı               : [{gri_matris.min()}, {gri_matris.max()}]")
    print(f"[+] Gri Piksel Ortalaması            : {gri_matris.mean():.3f}")

    rapor_basligi_yazdir("AŞAMA 5: Kapsamlı İstatistiksel Analiz Raporu")
    ozet = analizor.istatistikleri_hesapla()
    print(f"Genel Çözünürlük     : {ozet.genislik}x{ozet.yukseklik} ({ozet.toplam_piksel:,} piksel)")
    print(f"Bellek Tüketimi      : {ozet.bellek_kullanimi_kb:.2f} KB")
    print(f"Genel Piksel Ort.    : {ozet.genel_ortalama:.3f}")
    print(f"Genel Varyans        : {ozet.genel_varyans:.3f}")
    print("-" * 65)
    print(f"{'Kanal':<10} | {'Min':<5} | {'Max':<5} | {'Ortalama':<10} | {'Medyan':<8} | {'Std Sapma':<10}")
    print("-" * 65)
    for kanal_adi, ist in ozet.kanallar.items():
        print(
            f"{kanal_adi:<10} | {ist.en_kucuk:<5.0f} | {ist.en_buyuk:<5.0f} | "
            f"{ist.ortalama:<10.2f} | {ist.medyan:<8.1f} | {ist.standart_sapma:<10.2f}"
        )
    print("-" * 65)

    rapor_basligi_yazdir("AŞAMA 6: Piksel Değer Normalizasyonu Deneyleri")
    # 1. Min-Max [0.0, 1.0]
    min_max_0_1 = analizor.min_max_normallestir(hedef_aralik=(0.0, 1.0))
    print(
        f"[+] Min-Max [0, 1]  -> Min: {min_max_0_1.min():.4f}, Max: {min_max_0_1.max():.4f}, "
        f"Veri Tipi: {min_max_0_1.dtype}"
    )

    # 2. Min-Max [-1.0, 1.0] (Derin Öğrenme / Difüzyon Modelleri İçin)
    min_max_eksi1_1 = analizor.min_max_normallestir(hedef_aralik=(-1.0, 1.0))
    print(
        f"[+] Min-Max [-1, 1] -> Min: {min_max_eksi1_1.min():.4f}, Max: {min_max_eksi1_1.max():.4f}, "
        f"Veri Tipi: {min_max_eksi1_1.dtype}"
    )

    # 3. Z-Skoru (Kanal Bazlı Standartlaştırma: N(0, 1))
    z_skoru = analizor.z_skoru_normallestir(kanal_bazli=True)
    print("[+] Z-Skoru Standartlaştırması (Kanal Bazlı):")
    for idx, knl in enumerate(["Kırmızı", "Yeşil", "Mavi"]):
        knl_z = z_skoru[:, :, idx]
        print(
            f"    - {knl} Kanalı Z-Dağılımı -> Ortalama (mu): {knl_z.mean():>7.4f} (Hedef ~0), "
            f"Std Sapma (sigma): {knl_z.std():>6.4f} (Hedef ~1)"
        )

    rapor_basligi_yazdir("AŞAMA 7: Sayısal Taşma (Overflow) Koruması Testi")
    # 8-bit uint8 üzerinde 200 + 100 işlemi doğrudan yapılırsa 300 % 256 = 44 olur (taşma hatası!).
    # Geliştirdiğimiz güvenli fonksiyon bu hatayı engeller ve 255'e sınırlar (clipping).
    parlak_gorsel = analizor.parlaklik_ayarla(katsayi=1.5)
    print(f"[+] Orijinal Max Piksel Değeri      : {sentetik_gorsel.max()}")
    print(f"[+] 1.5x Parlaklık Sonrası Max Değer: {parlak_gorsel.max()} (Taşma engellendi, 255'te sınırlandı)")
    print("\n[V] Day 1: NumPy Görüntü Analizörü başarıyla icra edildi.")


if __name__ == "__main__":
    main()
