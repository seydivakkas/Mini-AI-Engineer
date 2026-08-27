"""Günün Ana Çalıştırma Akışı: GrabCut ile Ön Plan Ayırma ve Arka Plan Değiştirme.

Bu betik; karmaşık dokulu ve çok renkli zemin üzerinde duran dekoratif bir vazo/çömlek
sahnesi simüle eder, GrabCut (GMM + Graph Cut) ile ön planı arka plandan ayırır,
fırça darbeleriyle kenarları iyileştirir, şeffaf PNG ve yeni stüdyo kompoziti oluşturur.
"""

import sys
import time
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.grabcut_ayristirici import GrabCutAyristirici
from src.gorsellestirici import GrabCutGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 76
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def karmasik_urun_sahnesi_uret() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Karmaşık dokulu arka plan ve ortada dekoratif seramik vazo içeren sahne üretir."""
    h, w = 360, 360
    # 1. Karmaşık Dokulu Zemin (Zikzak şeritler ve renk dalgalanmaları)
    sahne = np.full((h, w, 3), (85, 95, 110), dtype=np.uint8)
    for i in range(0, w, 20):
        cv2.line(sahne, (i, 0), (i + 50, h), (70, 75, 90), 2)
        cv2.line(sahne, (w - i, 0), (w - i - 50, h), (110, 115, 130), 2)

    # 2. Ön Plan Nesnesi: Dekoratif Seramik Vazo (Merkez: 180, 190)
    # Gövde (Kiremit Kırmızısı: BGR 35, 60, 200)
    cv2.ellipse(sahne, (180, 210), (65, 85), 0, 0, 360, (35, 60, 200), -1)
    # Boyun
    cv2.rectangle(sahne, (155, 90), (205, 145), (35, 60, 200), -1)
    # Ağız
    cv2.ellipse(sahne, (180, 90), (35, 14), 0, 0, 360, (30, 50, 175), -1)
    # Taban
    cv2.ellipse(sahne, (180, 290), (45, 10), 0, 0, 360, (25, 45, 160), -1)

    # Vazo Kulpları (Sol ve Sağ Kavis)
    cv2.ellipse(sahne, (115, 160), (18, 38), 15, 0, 360, (35, 60, 200), 8)
    cv2.ellipse(sahne, (245, 160), (18, 38), -15, 0, 360, (35, 60, 200), 8)

    # Vazo Göbeğindeki Altın Desen Çizgisi (BGR: 30, 200, 240)
    cv2.ellipse(sahne, (180, 210), (60, 16), 0, 0, 360, (30, 200, 240), -1)

    # Nesneyi çevreleyen kullanıcı başlangıç sınırlayıcı kutusu
    dikdortgen = (85, 75, 190, 230)  # (x, y, w, h)
    return sahne, dikdortgen


def yeni_studyo_arka_plani_uret(h: int, w: int) -> np.ndarray:
    """Modern degrade stüdyo arka planı üretir (Koyu lacivert - antrasit geçişi)."""
    arka_plan = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        oran = y / float(h)
        # Üst: Koyu Petrol Mavisi (80, 50, 20), Alt: Açık Duman Gri (160, 160, 170)
        b = int(80 * (1 - oran) + 160 * oran)
        g = int(50 * (1 - oran) + 160 * oran)
        r = int(20 * (1 - oran) + 170 * oran)
        arka_plan[y, :] = (b, g, r)
    return arka_plan


def main() -> None:
    baslik("AŞAMA 1: Karmaşık Sahnenin ve Başlangıç Kutusunun Hazırlanması")
    sahne_bgr, dikdortgen = karmasik_urun_sahnesi_uret()
    h, w = sahne_bgr.shape[:2]
    toplam_piksel = h * w
    print(f"[+] Görüntü Çözünürlüğü       : {w} x {h} ({toplam_piksel:,} piksel)")
    print(f"[+] Başlangıç Dikdörtgeni     : (x={dikdortgen[0]}, y={dikdortgen[1]}, w={dikdortgen[2]}, h={dikdortgen[3]})")
    print(f"[+] Kutu Alanı                : {dikdortgen[2] * dikdortgen[3]:,} piksel (%{(dikdortgen[2] * dikdortgen[3] / toplam_piksel) * 100:.1f})")

    baslik("AŞAMA 2: GrabCut (GMM + Graph Cut) 1. Faz: Dikdörtgen Başlatma")
    ayristirici = GrabCutAyristirici()
    baslangic_zamani = time.perf_counter()

    on_plan_1, ikili_maske_1, ham_maske_1 = ayristirici.dikdortgen_ile_ayristir(
        sahne_bgr, dikdortgen, iterasyon_sayisi=5
    )
    sure_faz1 = time.perf_counter() - baslangic_zamani
    on_plan_piksel_1 = int(np.sum(ikili_maske_1 > 0))
    print(f"[V] Faz 1 Tamamlandı (5 İterasyon): {sure_faz1:.3f} saniye")
    print(f"[V] Ayrıştırılan Ön Plan Pikselleri: {on_plan_piksel_1:,} adet (%{(on_plan_piksel_1 / toplam_piksel) * 100:.1f})")

    baslik("AŞAMA 3: GrabCut 2. Faz: İnteraktif Fırça İpuçlarıyla Kenar İyileştirme")
    # Kulp içindeki arka plan boşluğunu kesin arka plan (GC_BGD) olarak işaretle
    kesin_bg_noktalari = [(115, 160), (245, 160)]  # Kulp iç delikleri
    kesin_fg_noktalari = [(180, 95), (180, 285)]   # Ağız ve taban noktaları

    # Fırça görselleştirmesi için kopya
    firca_katmani = np.zeros_like(sahne_bgr)
    for pt in kesin_bg_noktalari:
        cv2.circle(firca_katmani, pt, 5, (0, 0, 255), -1)  # Kırmızı: Arka Plan
    for pt in kesin_fg_noktalari:
        cv2.circle(firca_katmani, pt, 5, (0, 255, 0), -1)  # Yeşil: Ön Plan

    on_plan_2, ikili_maske_2, ham_maske_2 = ayristirici.maske_ile_iyilestir(
        gorsel_bgr=sahne_bgr,
        mevcut_ham_maske=ham_maske_1,
        kesin_on_plan_noktalari=kesin_fg_noktalari,
        kesin_arka_plan_noktalari=kesin_bg_noktalari,
        firca_yaricapi=4,
        iterasyon_sayisi=3
    )
    on_plan_piksel_2 = int(np.sum(ikili_maske_2 > 0))
    print(f"[V] Faz 2 Tamamlandı (İnteraktif Maske İyileştirme)")
    print(f"[V] İyileştirilmiş Ön Plan Pikselleri: {on_plan_piksel_2:,} adet")

    baslik("AŞAMA 4: Şeffaf PNG (RGBA) ve Yeni Stüdyo Arka Plan Kompoziti")
    seffaf_bgra = ayristirici.seffaf_png_olustur(sahne_bgr, ikili_maske_2)
    studyo_arkaplan = yeni_studyo_arka_plani_uret(h, w)
    kompozit = ayristirici.arka_plan_degistir(sahne_bgr, ikili_maske_2, studyo_arkaplan, kenar_yumusatma_yaricap=2)

    cikti_klasoru = proje_kok / "ciktilar"
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    seffaf_dosya = cikti_klasoru / "izole_nesne_seffaf.png"
    cv2.imwrite(str(seffaf_dosya), seffaf_bgra)
    print(f"[V] Şeffaf 4 kanallı (BGRA) ürün görseli kaydedildi: {seffaf_dosya.name}")

    baslik("AŞAMA 5: 4 Panelli Analiz Çizelgesinin Kaydedilmesi")
    rapor_dosya = cikti_klasoru / "grabcut_segmentasyon_paneli.png"
    kaydedilen = GrabCutGorsellestirici.analiz_paneli_ciz(
        orijinal_bgr=sahne_bgr,
        dikdortgen=dikdortgen,
        ham_maske=ham_maske_2,
        izole_on_plan_bgr=on_plan_2,
        kompozit_bgr=kompozit,
        dosya_yolu=rapor_dosya,
        firca_izleri_bgr=firca_katmani
    )
    print(f"[V] GrabCut analiz paneli başarıyla kaydedildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 15: GrabCut Ön Plan ve Arka Plan Segmentasyonu başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
