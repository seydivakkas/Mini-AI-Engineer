"""Günün Ana Çalıştırma Akışı: Motif Segmentasyonu ve Kontur Ayrıştırma.

Bu betik; farklı geometrik şekiller içeren sentetik geleneksel çini/kilim sahnesi
üretir, Otsu eşikleme ve morfolojik filtrelerle motifleri arka plandan ayırır,
her motif için alan, çevre, dairesellik, doluluk (solidity) ve sınırlayıcı kutuları
(Bounding Box) hesaplayarak 4 panelli analiz çizelgesini kaydeder.
"""

import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.motif_ayristirici import MotifAyristirici
from src.gorsellestirici import MotifGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 76
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def sentetik_cini_motifi_sahnesi() -> np.ndarray:
    """Farklı geometrik motifler içeren 420 x 420 sentetik çini/kilim sahnesi üretir."""
    sahne = np.full((420, 420, 3), (40, 25, 20), dtype=np.uint8)  # Koyu lacivert zemin

    # 1. Merkez: Sekiz Köşeli Yıldız / Madalyon Motifi (Altın Sarısı)
    merkez_x, merkez_y = 210, 210
    kare_1 = np.array([
        [merkez_x - 45, merkez_y - 45], [merkez_x + 45, merkez_y - 45],
        [merkez_x + 45, merkez_y + 45], [merkez_x - 45, merkez_y + 45]
    ], dtype=np.int32)
    kare_2 = np.array([
        [merkez_x, merkez_y - 64], [merkez_x + 64, merkez_y],
        [merkez_x, merkez_y + 64], [merkez_x - 64, merkez_y]
    ], dtype=np.int32)
    cv2.fillPoly(sahne, [kare_1], (30, 200, 240))
    cv2.fillPoly(sahne, [kare_2], (30, 200, 240))

    # 2. Sol-Üst: Baklava / Eşkenar Dörtgen (Kiremit Kırmızısı)
    baklava = np.array([
        [90, 40], [140, 90], [90, 140], [40, 90]
    ], dtype=np.int32)
    cv2.fillPoly(sahne, [baklava], (40, 50, 210))

    # 3. Sağ-Üst: Dairesel Rozet Çiçek (Zümrüt Yeşili)
    cv2.circle(sahne, (330, 90), 45, (50, 180, 70), -1)

    # 4. Sol-Alt: Elibelinde / Üçgen Geometrik Motif (Krem)
    ucgen = np.array([
        [90, 270], [140, 370], [40, 370]
    ], dtype=np.int32)
    cv2.fillPoly(sahne, [ucgen], (220, 235, 245))

    # 5. Sağ-Alt: Yatay Oval / Elips Rozet (Hardal Sarısı)
    cv2.ellipse(sahne, (330, 320), (55, 35), 25, 0, 360, (30, 160, 220), -1)

    # Gerçekçi doku için hafif Gauss gürültüsü
    gurultu = np.random.normal(0, 4, sahne.shape).astype(np.int16)
    sahne_gurultulu = np.clip(sahne.astype(np.int16) + gurultu, 0, 255).astype(np.uint8)

    return sahne_gurultulu


def main() -> None:
    baslik("AŞAMA 1: Sentetik Motif Sahnesinin Oluşturulması")
    sahne_bgr = sentetik_cini_motifi_sahnesi()
    h, w, c = sahne_bgr.shape
    print(f"[+] Sahne Çözünürlüğü         : {w} x {h} piksel ({c} kanal)")
    print("[+] Sahne İçeriği             : 5 Farklı Geometrik Anadolu Motifi")
    print("    * Yıldız Madalyon (Merkez)")
    print("    * Baklava Eşkenar Dörtgen (Sol-Üst)")
    print("    * Dairesel Rozet (Sağ-Üst)")
    print("    * Geometrik Üçgen (Sol-Alt)")
    print("    * Açılı Elips Rozet (Sağ-Alt)")

    baslik("AŞAMA 2: Otsu Eşikleme ve Morfolojik Filtreleme")
    gri = cv2.cvtColor(sahne_bgr, cv2.COLOR_BGR2GRAY)
    _, esik_degeri = MotifAyristirici.otsu_esikleme(gri)
    print(f"[V] Otsu Optimal Eşik Değeri (T*): {esik_degeri:.1f}")

    baslik("AŞAMA 3: Kontur Ayrıştırma ve Şekil Analitiği")
    motifler, ikili_maske = MotifAyristirici.motifleri_ayristir(
        sahne_bgr, min_alan=300.0, maks_alan_orani=0.80
    )
    print(f"[V] Tespit Edilen Geçerli Motif Adedi: {len(motifler)}")

    print("-" * 76)
    print(f"{'ID':<5} | {'Alan (px)':<10} | {'Çevre':<8} | {'Dairesellik':<12} | {'Solidity':<10} | {'Sınırlayıcı Kutu (x,y,w,h)'}")
    print("-" * 76)
    for m in motifler:
        kutu_str = f"({m.sinirlayici_kutu[0]}, {m.sinirlayici_kutu[1]}, {m.sinirlayici_kutu[2]}, {m.sinirlayici_kutu[3]})"
        print(f"M-{m.motif_id:<3} | {m.alan:<10.0f} | {m.cevre:<8.1f} | {m.dairesellik:<12.3f} | {m.doluluk_orani:<10.3f} | {kutu_str}")
    print("-" * 76)

    # Şekil analizi yorumu
    en_dairesel = max(motifler, key=lambda m: m.dairesellik)
    en_girintili = min(motifler, key=lambda m: m.doluluk_orani)
    print(f"[+] En Yüksek Dairesellik : Motif M-{en_dairesel.motif_id} (Skor: {en_dairesel.dairesellik:.3f} - Dairesel Rozet!)")
    print(f"[+] En Girintili Motif    : Motif M-{en_girintili.motif_id} (Solidity: {en_girintili.doluluk_orani:.3f} - Köşeli Yıldız!)")

    baslik("AŞAMA 4: 4 Panelli Görsel Raporun ve Kırpılmış Galerinin Kaydedilmesi")
    cikti_yolu = proje_kok / "ciktilar" / "motif_segmentasyon_paneli.png"
    kaydedilen = MotifGorsellestirici.analiz_paneli_ciz(
        orijinal_bgr=sahne_bgr,
        ikili_maske=ikili_maske,
        motifler=motifler,
        dosya_yolu=cikti_yolu
    )
    print(f"[V] Motif analiz paneli başarıyla kaydedildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 14: Görsellerdeki Desen ve Motiflerin Ayrıştırılması başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
