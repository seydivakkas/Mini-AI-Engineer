"""Günün Ana Çalıştırma Akışı: Geleneksel Görsel Öznitelik Çıkarımı Karşılaştırması.

Bu betik; zengin dokulu ve geometrik desenli bir test görüntüsü üzerinde
SIFT, ORB, HOG ve LBP algoritmalarını çalıştırır, boyut, bellek ve işlem süresi
metriklerini karşılaştırmalı bir tablo halinde sunar ve 4 panelli analiz çizelgesini kaydeder.
"""

import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.oznitelik_cikarici import GorselOznitelikCikarici, OznitelikOzeti
from src.gorsellestirici import OznitelikGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 78
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def zengin_test_gorseli_uret() -> np.ndarray:
    """SIFT, ORB, HOG ve LBP için zengin köşe, kenar ve mikro doku içeren 320x320 gri görsel üretir."""
    h, w = 320, 320
    # 1. Taban: İnce Kumaş Dokusu Simülasyonu (LBP için mikro desen)
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    doku = (np.sin(xx * 0.3) * np.cos(yy * 0.3) * 20 + 128).astype(np.uint8)

    # 2. Köşeler ve Şekiller (SIFT ve ORB için)
    # Büyük Baklava Dilimi
    baklava = np.array([[160, 40], [260, 160], [160, 280], [60, 160]], dtype=np.int32)
    cv2.fillPoly(doku, [baklava], 40)
    cv2.polylines(doku, [baklava], isClosed=True, color=240, thickness=3)

    # İç Sekizgen Yıldız
    yildiz = np.array([
        [160, 100], [180, 140], [220, 160], [180, 180],
        [160, 220], [140, 180], [100, 160], [140, 140]
    ], dtype=np.int32)
    cv2.fillPoly(doku, [yildiz], 220)

    # Daireler ve Çerçeveler (HOG kenar yönelimleri için)
    cv2.circle(doku, (160, 160), 25, 30, 3)
    cv2.rectangle(doku, (15, 15), (w - 15, h - 15), 230, 3)

    # 4 Köşede Küçük Doku Rozetleri
    for cx, cy in [(40, 40), (280, 40), (40, 280), (280, 280)]:
        cv2.circle(doku, (cx, cy), 15, 200, -1)
        cv2.circle(doku, (cx, cy), 8, 40, -1)

    return doku


def main() -> None:
    baslik("AŞAMA 1: Zengin Geometrik ve Dokusal Test Görselinin Üretilmesi")
    gorsel_gri = zengin_test_gorseli_uret()
    h, w = gorsel_gri.shape
    print(f"[+] Görüntü Çözünürlüğü         : {w} x {h} piksel (Tek Kanal - Gri)")
    print("[+] Test Sahnesi Öğeleri        : Kumaş mikro dokusu, sekizgen yıldız, baklava ve daireler")

    # Isınma (Warmup) geçişi - Python/C++ iç arabellekleri hazırla
    _ = GorselOznitelikCikarici.sift_cikar(gorsel_gri, maks_nokta=50)
    _ = GorselOznitelikCikarici.orb_cikar(gorsel_gri, maks_nokta=50)

    baslik("AŞAMA 2: SIFT, ORB, HOG ve LBP Algoritmalarının Koşturulması")

    # 1. SIFT
    sift_kp, sift_des, sift_sure = GorselOznitelikCikarici.sift_cikar(gorsel_gri, maks_nokta=500)
    ozet_sift = OznitelikOzeti(
        algoritma="SIFT",
        anahtar_nokta_sayisi=len(sift_kp),
        tanimlayici_boyutu=sift_des.shape,
        veri_tipi=str(sift_des.dtype),
        bellek_bayt=sift_des.nbytes,
        calisma_suresi_ms=sift_sure,
        aciklama="Ölçek & Açı Bağımsız (128-B Float)"
    )

    # 2. ORB
    orb_kp, orb_des, orb_sure = GorselOznitelikCikarici.orb_cikar(gorsel_gri, maks_nokta=500)
    ozet_orb = OznitelikOzeti(
        algoritma="ORB",
        anahtar_nokta_sayisi=len(orb_kp),
        tanimlayici_boyutu=orb_des.shape,
        veri_tipi=str(orb_des.dtype),
        bellek_bayt=orb_des.nbytes,
        calisma_suresi_ms=orb_sure,
        aciklama="Hızlı İkili Hamming (32 Byte / 256-Bit)"
    )

    # 3. HOG
    hog_vektor, hog_harita, hog_sure = GorselOznitelikCikarici.hog_cikar(gorsel_gri)
    ozet_hog = OznitelikOzeti(
        algoritma="HOG",
        anahtar_nokta_sayisi=0,
        tanimlayici_boyutu=hog_vektor.shape,
        veri_tipi=str(hog_vektor.dtype),
        bellek_bayt=hog_vektor.nbytes,
        calisma_suresi_ms=hog_sure,
        aciklama="Yoğun Gradyan Dağılımı (Nesne Şekli)"
    )

    # 4. LBP
    lbp_harita, lbp_hist, lbp_sure = GorselOznitelikCikarici.lbp_cikar(gorsel_gri)
    ozet_lbp = OznitelikOzeti(
        algoritma="LBP",
        anahtar_nokta_sayisi=0,
        tanimlayici_boyutu=lbp_hist.shape,
        veri_tipi=str(lbp_hist.dtype),
        bellek_bayt=lbp_hist.nbytes,
        calisma_suresi_ms=lbp_sure,
        aciklama="Yerel İkili Doku Parmak İzi (Uniform)"
    )

    ozetler = [ozet_sift, ozet_orb, ozet_hog, ozet_lbp]

    baslik("AŞAMA 3: Algoritmik Karşılaştırma ve Telemetri Raporu")
    print(f"{'Algoritma':<10} | {'Nokta Adedi':<12} | {'Çıktı Boyutu':<16} | {'Veri Tipi':<10} | {'Bellek':<10} | {'Süre (ms)':<10}")
    print("-" * 78)
    for o in ozetler:
        nokta_str = f"{o.anahtar_nokta_sayisi:,}" if o.anahtar_nokta_sayisi > 0 else "Yoğun (Dense)"
        boyut_str = str(o.tanimlayici_boyutu)
        bellek_str = f"{o.bellek_bayt:,} B"
        print(f"{o.algoritma:<10} | {nokta_str:<12} | {boyut_str:<16} | {o.veri_tipi:<10} | {bellek_str:<10} | {o.calisma_suresi_ms:<10.2f}")
    print("-" * 78)

    # Mühendislik Analizi
    hiz_katsayisi = ozet_sift.calisma_suresi_ms / max(0.01, ozet_orb.calisma_suresi_ms)
    bellek_tasarrufu = (ozet_sift.bellek_bayt / max(1, ozet_orb.bellek_bayt))
    print(f"\n[+] Mühendislik Çıkarımları:")
    print(f"    * ORB, SIFT'e kıyasla yaklaşık {hiz_katsayisi:.1f}x kat DAHA HIZLI çalıştı.")
    print(f"    * ORB tanımlayıcıları, SIFT'e göre {bellek_tasarrufu:.1f}x kat DAHA AZ RAM tüketti (Binary Hamming!).")
    print(f"    * HOG {len(hog_vektor):,} boyutlu global şekil vektörü üretti (SVM sınıflandırıcıları için ideal).")
    print(f"    * LBP yalnızca {len(lbp_hist)} boyutlu kompakt bir doku histogramı ile yüzeyi özetledi.")

    baslik("AŞAMA 4: 4 Panelli Analiz Çizelgesinin Kaydedilmesi")
    cikti_klasoru = proje_kok / "ciktilar"
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    rapor_dosyasi = cikti_klasoru / "oznitelik_analiz_paneli.png"

    kaydedilen = OznitelikGorsellestirici.analiz_paneli_ciz(
        gorsel_gri=gorsel_gri,
        sift_kp=sift_kp,
        orb_kp=orb_kp,
        hog_harita=hog_harita,
        lbp_harita=lbp_harita,
        lbp_hist=lbp_hist,
        dosya_yolu=rapor_dosyasi
    )
    print(f"[V] Öznitelik analiz paneli kaydedildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 16: Geleneksel Görsel Öznitelik Çıkarımı başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
