"""Günün Ana Çalıştırma Akışı: Perspektif Düzeltme ve Kuşbakışı Dönüşümü.

Bu betik; ahşap parke zemin üzerinde açılı duran perspektif bozulmalı bir halı
deseni simüle eder, 4 köşe noktasını sıralar, homografi matrisini çözer,
nesneyi tam ortogonal kuşbakışı düzleme taşır ve 3 panelli analiz raporunu kaydeder.
"""

import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.perspektif_duzeltici import PerspektifDuzeltici
from src.gorsellestirici import PerspektifGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def perspektifli_sahne_uret() -> tuple[np.ndarray, np.ndarray]:
    """Zemin üzerinde açılı/çarpık duran sentetik halı sahnesi ve gerçek köşe koordinatlarını üretir."""
    sahne_h, sahne_w = 400, 450

    # 1. Ahşap parke dokulu zemin (Gri-kahve şeritler)
    zemin = np.full((sahne_h, sahne_w, 3), (70, 80, 95), dtype=np.uint8)
    for y in range(0, sahne_h, 30):
        cv2.line(zemin, (0, y), (sahne_w, y), (50, 60, 75), 1)

    # 2. Düzgün Kuşbakışı Referans Halı Deseni (200 x 300 piksel)
    referans_w, referans_h = 220, 320
    duz_hali = np.full((referans_h, referans_w, 3), (30, 40, 180), dtype=np.uint8)  # Kiremit kırmızısı

    # Kenarlıklar ve motifler
    cv2.rectangle(duz_hali, (15, 15), (referans_w - 15, referans_h - 15), (30, 190, 230), 4)  # Altın sarısı çerçeve
    cv2.circle(duz_hali, (referans_w // 2, referans_h // 2), 60, (50, 130, 40), -1)  # Yeşil merkez madalyon
    cv2.circle(duz_hali, (referans_w // 2, referans_h // 2), 30, (230, 240, 245), -1)  # Krem iç göbek
    cv2.putText(
        duz_hali, "AI MOTIF", (referans_w // 2 - 45, referans_h // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 30, 80), 2
    )

    # 3. Halıyı zemine açılı perspektifle yerleştir (3D izdüşüm)
    kaynak_koseler = np.array([
        [0, 0],
        [referans_w - 1, 0],
        [referans_w - 1, referans_h - 1],
        [0, referans_h - 1]
    ], dtype=np.float32)

    # Zemindeki bozulmuş açılı hedef koordinatlar (Yamuk / Quadrilateral)
    zemindeki_koseler = np.array([
        [90.0, 60.0],    # Sol-Üst (uzakta, daha dar)
        [340.0, 95.0],   # Sağ-Üst
        [390.0, 360.0],  # Sağ-Alt (yakında, daha geniş)
        [45.0, 325.0]    # Sol-Alt
    ], dtype=np.float32)

    # Düz halıyı zemindeki yamuğa projekte et
    H_ileri = cv2.getPerspectiveTransform(kaynak_koseler, zemindeki_koseler)
    acili_hali = cv2.warpPerspective(duz_hali, H_ileri, (sahne_w, sahne_h))

    # Maske ile halıyı zemine yapıştır
    maske = (acili_hali > 0).any(axis=2)
    zemin[maske] = acili_hali[maske]

    return zemin, zemindeki_koseler


def main() -> None:
    baslik("AŞAMA 1: Perspektif Bozulmalı Sentetik Sahnenin İncelenmesi")
    sahne_bgr, bozuk_noktalar = perspektifli_sahne_uret()
    h, w = sahne_bgr.shape[:2]
    print(f"[+] Sahne Çözünürlüğü         : {w} x {h} piksel")
    print("[+] Tespit Edilen Bozuk Köşeler (Rastgele Giriş):")
    for i, pt in enumerate(bozuk_noktalar, 1):
        print(f"    * Nokta #{i}: (x={pt[0]:.1f}, y={pt[1]:.1f})")

    baslik("AŞAMA 2: 4 Köşe Noktasının Saat Yönünde Matematiksel Sıralanması")
    # Bilerek sırayı karıştırarak sıralama fonksiyonunu test edelim
    karisik_noktalar = bozuk_noktalar[[2, 0, 3, 1]]
    sirali_noktalar = PerspektifDuzeltici.noktalari_sirala(karisik_noktalar)

    etiketler = ["Sol-Üst (Top-Left)", "Sağ-Üst (Top-Right)", "Sağ-Alt (Bottom-Right)", "Sol-Alt (Bottom-Left)"]
    for etiket, pt in zip(etiketler, sirali_noktalar):
        print(f"[V] {etiket:<25} -> (x={pt[0]:.1f}, y={pt[1]:.1f})")

    baslik("AŞAMA 3: Hedef Kuşbakışı Çözünürlüğünün Hesaplanması")
    hedef_w, hedef_h = PerspektifDuzeltici.hedef_boyutlari_hesapla(sirali_noktalar)
    print(f"[+] Hesaplanmış İdeal Genişlik (W) : {hedef_w} piksel")
    print(f"[+] Hesaplanmış İdeal Yükseklik (H): {hedef_h} piksel")
    print(f"[+] En/Boy Oranı (Aspect Ratio)    : {hedef_w / hedef_h:.3f}")

    baslik("AŞAMA 4: 3x3 Homografi Matrisinin Çözümü ve Görüntü Dönüşümü")
    duzeltilmis, H_matrisi = PerspektifDuzeltici.dort_nokta_donusumu(
        sahne_bgr,
        sirali_noktalar,
        hedef_genislik=hedef_w,
        hedef_yukseklik=hedef_h
    )

    print("[+] 3x3 Homografi Projeksiyon Matrisi (H):")
    for satir in H_matrisi:
        print("    [ " + "  ".join(f"{val:12.4e}" for val in satir) + " ]")

    det_h = np.linalg.det(H_matrisi)
    print(f"[V] Homografi Determinantı: {det_h:.4e} (Tersinir ve geçerli dönüşüm!)")
    print(f"[V] Düzeltilmiş Çıktı Boyutu: {duzeltilmis.shape[1]} x {duzeltilmis.shape[0]} px")

    baslik("AŞAMA 5: Analiz Panelinin ve Isı Haritasının Kaydedilmesi")
    cikti_yolu = proje_kok / "ciktilar" / "perspektif_duzeltme_paneli.png"
    kaydedilen = PerspektifGorsellestirici.analiz_paneli_ciz(
        orijinal_bgr=sahne_bgr,
        sirali_noktalar=sirali_noktalar,
        duzeltilmis_bgr=duzeltilmis,
        homografi_matrisi=H_matrisi,
        dosya_yolu=cikti_yolu
    )
    print(f"[V] Perspektif analiz paneli kaydedildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 13: Geometrik Dönüşümler ve Perspektif Düzeltme başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
