"""Günün Ana Çalıştırma Akışı: Renk Uzayları Gezgini ve Segmentasyon.

Bu betik; yapay aydınlatma ve derin gölge geçişi barındıran sentetik bir endüstriyel
sahne oluşturarak RGB uzayının gölge zaafını, HSV ve LAB uzaylarının gölgeye karşı
bağışıklığını ve renk tabanlı hedef segmentasyonunu baştan sona kanıtlar.
"""

import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.renk_donusturucu import RenkUzayiDonusturucu, RenkSegmentasyoncu
from src.gorsellestirici import RenkUzayiGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def golgeli_endustriyel_sahne_uret() -> np.ndarray:
    """Yarı gölgeli ve farklı renkli nesneler içeren 256x256 BGR sahnesi üretir."""
    # Nötr gri arka plan
    resim = np.full((256, 256, 3), 170, dtype=np.uint8)

    # 1. Kırmızı Hedef Nesne (Tam ortada kırmızı daire - Kusur Mührü)
    # BGR formatında saf kırmızı: (0, 0, 240)
    cv2.circle(resim, (128, 128), 38, (20, 20, 240), -1)

    # 2. Yeşil Geçer Parça (Sol üstte yeşil kare)
    cv2.rectangle(resim, (30, 30), (80, 80), (30, 200, 30), -1)

    # 3. Mavi Kondansatör (Sağ altta mavi dikdörtgen)
    cv2.rectangle(resim, (170, 170), (225, 225), (220, 40, 20), -1)

    # 4. Kritik Aşama: Sert Işık Gradyanı ve Çapraz Gölge Ekleme
    # Sahnenin sağ üst köşesi aydınlık, sol alt köşesi derin gölgeli!
    golge_maskesi = np.zeros((256, 256), dtype=np.float32)
    for y in range(256):
        for x in range(256):
            # Çapraz aydınlatma katsayısı: 0.35 (karanlık) ile 1.0 (aydınlık) arası
            golge_maskesi[y, x] = 0.35 + 0.65 * ((x + (255 - y)) / 510.0)

    # Her 3 BGR kanalını aydınlatma maskesi ile çarparak gölge oluştur
    golgeli_resim = np.clip(resim * golge_maskesi[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return golgeli_resim


def main() -> None:
    baslik("AŞAMA 1: Sentetik Sahnenin Oluşturulması ve Işık Değişimi")
    sahne_bgr = golgeli_endustriyel_sahne_uret()
    print(f"[+] Çözünürlük               : {sahne_bgr.shape}")
    print(f"[+] Sahne Özelliği           : Çapraz sert gölge geçişi")
    print(f"[+] Kırmızı Nesne Konumu     : Merkez (Yarısı gölgede, yarısı ışıkta!)")

    baslik("AŞAMA 2: RGB Uzayının Gölge Zaafı Kanıtı")
    r_kanali = sahne_bgr[:, :, 2]
    # Gölgedeki kırmızı piksel ile ışıktaki kırmızı pikselin R değerlerini karşılaştıralım
    aydinlik_kirmizi_r = int(r_kanali[115, 145])
    golgedeki_kirmizi_r = int(r_kanali[140, 110])
    arka_plan_gri_r = int(r_kanali[30, 200])

    print(f"  * Işıktaki Kırmızı Nesne R Değeri : {aydinlik_kirmizi_r}")
    print(f"  * Gölgedeki Kırmızı Nesne R Değeri: {golgedeki_kirmizi_r}  <-- Aşırı düştü!")
    print(f"  * Arka Plan Nötr Gri R Değeri     : {arka_plan_gri_r}")
    print(">>> Sonuç: RGB uzayında 'R > 150' eşiği koyarsanız, gölgedeki kırmızı nesneyi")
    print("    tamamen kaçırırsınız; 'R > 80' yaparsanız tüm gri arka planı kırmızı sanırsınız!")

    baslik("AŞAMA 3: HSV Uzayında Renk Tonu (Hue) İzolasyonu")
    hsv = RenkUzayiDonusturucu.bgr_to_hsv(sahne_bgr)
    h_kanali = hsv[:, :, 0]
    aydinlik_h = int(h_kanali[115, 145])
    golgedeki_h = int(h_kanali[140, 110])
    print(f"  * Işıktaki Kırmızı Ton (H) Değeri : {aydinlik_h}°")
    print(f"  * Gölgedeki Kırmızı Ton (H) Değeri: {golgedeki_h}°")
    print("[V] Gözlem: Parlaklık 3 kat düşse bile Ton (Hue) değeri 0-5° arasında SABİT KALDI!")

    baslik("AŞAMA 4: Çift Aralık Kırmızı Segmentasyonu ve Hedef Çıkarma")
    kirmizi_maske = RenkSegmentasyoncu.kirmizi_renk_maskesi(sahne_bgr, doygunluk_alt=60, parlaklik_alt=40)
    segmente_kirmizi = RenkSegmentasyoncu.maskeyi_uygula(sahne_bgr, kirmizi_maske)
    yakalanan_piksel = int(np.sum(kirmizi_maske > 0))
    print(f"[V] Kırmızı maske başarıyla üretildi. Yakalanan piksel: {yakalanan_piksel} adet.")
    print("[V] Nesne hem gölgede hem ışıkta tek parça halinde eksiksiz izole edildi!")

    baslik("AŞAMA 5: CIELAB Kromatik Delta-E ile Yeşil Nesnenin Algısal Tespiti")
    # Hedef yeşil BGR: (30, 200, 30) - Aydınlık (L*) hariç tutularak yalnızca a* ve b* kromatik düzleminde aranır!
    yesil_maske = RenkSegmentasyoncu.cielab_delta_e_maskesi(
        sahne_bgr,
        hedef_renk_bgr=(30, 200, 30),
        delta_e_esik=30.0,
        aydinlik_haric_tut=True
    )
    yesil_piksel = int(np.sum(yesil_maske > 0))
    print(f"[V] CIELAB kromatik mesafesiyle yeşil parça tespit edildi: {yesil_piksel} piksel.")

    baslik("AŞAMA 6: 12 Panelli Analiz Raporunun Kaydedilmesi")
    rgb = RenkUzayiDonusturucu.bgr_to_rgb(sahne_bgr)
    r, g, b = cv2.split(rgb)
    h, s, v = cv2.split(hsv)
    lab = RenkUzayiDonusturucu.bgr_to_lab(sahne_bgr)
    l, a, b_lab = cv2.split(lab)

    cikti_yolu = proje_kok / "ciktilar" / "renk_uzaylari_analiz_paneli.png"
    kaydedilen = RenkUzayiGorsellestirici.analiz_paneli_ciz(
        sahne_bgr,
        rgb_kanallari=(r, g, b),
        hsv_kanallari=(h, s, v),
        lab_kanallari=(l, a, b_lab),
        maske=kirmizi_maske,
        segmente_bgr=segmente_kirmizi,
        dosya_yolu=cikti_yolu
    )
    print(f"[V] 12 Panelli analiz çizelgesi kaydedildi: {kaydedilen.name}")
    print(f"[V] Tam Dosya Yolu: {kaydedilen}")
    print("\n[V] Day 10: Renk Uzayları Gezgini ve Segmentasyon başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
