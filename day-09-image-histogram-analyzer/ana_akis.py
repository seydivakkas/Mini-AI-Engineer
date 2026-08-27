"""Günün Ana Çalıştırma Akışı: Görüntü Histogramı ve Kontrast İyileştirme.

Bu betik; düşük kontrastlı ve dar dinamik aralıklı sentetik bir endüstriyel görsel
üreterek Global Histogram Eşitleme ve CLAHE yöntemlerini kıyaslar,
entropi/kontrast metriklerini ölçer ve 2x3 karşılaştırma panelini diske kaydeder.
"""

import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.histogram_motoru import HistogramHesaplayici, KontrastIyilestirici
from src.gorsellestirici import HistogramGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def dusuk_kontrastli_test_gorseli_uret() -> np.ndarray:
    """X-ışını veya karanlık fabrika ortamını taklit eden dar aralıklı sentetik görsel üretir."""
    # Taban zemin: Çok dar aralık (40 - 70 arası)
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    xx, yy = np.meshgrid(x, y)
    dalga = 50 + 15 * np.sin(xx) * np.cos(yy)
    resim = dalga.astype(np.uint8)

    # Gizli iç detaylar (karanlıkta gömülü nesneler)
    cv2.circle(resim, (80, 80), 40, 75, -1)
    cv2.rectangle(resim, (150, 140), (220, 210), 65, -1)
    cv2.putText(resim, "AI LAB", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 80, 2)

    # Bir köşede aşırı parlak ışık parlaması (Glaring / Backlight)
    cv2.circle(resim, (230, 30), 25, 230, -1)

    return resim


def main() -> None:
    baslik("AŞAMA 1: Düşük Kontrastlı Sentetik Görselin İncelenmesi")
    ham_gorsel = dusuk_kontrastli_test_gorseli_uret()
    ham_metrikler = HistogramHesaplayici.kontrast_metrikleri(ham_gorsel)

    print(f"[+] Çözünürlük         : {ham_gorsel.shape}")
    print(f"[+] Min - Max Piksel   : [{ham_metrikler['min_piksel']}, {ham_metrikler['max_piksel']}]")
    print(f"[+] Dinamik Aralık     : {ham_metrikler['dinamik_aralik']}")
    print(f"[+] RMS Kontrast (Std) : {ham_metrikler['rms_kontrast']}")
    print(f"[+] Shannon Entropisi  : {ham_metrikler['shannon_entropisi']} bit")

    baslik("AŞAMA 2: Global Histogram Eşitleme (Global Equalization)")
    global_esitlenmis = KontrastIyilestirici.global_histogram_esitle(ham_gorsel)
    global_metrikler = HistogramHesaplayici.kontrast_metrikleri(global_esitlenmis)
    print("[V] Tüm görselin kümülatif olasılık fonksiyonu (CDF) doğrusal dağıtıldı.")
    print(f"[+] Yeni Dinamik Aralık: {global_metrikler['dinamik_aralik']}")
    print(f"[+] Yeni RMS Kontrast  : {global_metrikler['rms_kontrast']}")
    print(f"[+] Yeni Entropi       : {global_metrikler['shannon_entropisi']} bit")

    baslik("AŞAMA 3: CLAHE (Kontrast Sınırlı Uyarlanabilir Eşitleme)")
    clahe_gorsel = KontrastIyilestirici.clahe_uygula(ham_gorsel, kirpma_limiti=2.5, karo_boyutu=(8, 8))
    clahe_metrikler = HistogramHesaplayici.kontrast_metrikleri(clahe_gorsel)
    print("[V] Görüntü 8x8 karolara bölündü, yerel kontrast dengelendi ve parazit kırpıldı.")
    print(f"[+] Yeni Dinamik Aralık: {clahe_metrikler['dinamik_aralik']}")
    print(f"[+] Yeni RMS Kontrast  : {clahe_metrikler['rms_kontrast']}")
    print(f"[+] Yeni Entropi       : {clahe_metrikler['shannon_entropisi']} bit")

    baslik("AŞAMA 4: Karşılaştırmalı Metrik Tablosu")
    print(f"{'Yöntem Adı':<25} | {'Dinamik Aralık':<16} | {'RMS Kontrast':<14} | {'Shannon Entropisi'}")
    print("-" * 74)
    print(f"{'Ham Görsel':<25} | {ham_metrikler['dinamik_aralik']:<16} | {ham_metrikler['rms_kontrast']:<14} | {ham_metrikler['shannon_entropisi']} bit")
    print(f"{'Global Eşitleme':<25} | {global_metrikler['dinamik_aralik']:<16} | {global_metrikler['rms_kontrast']:<14} | {global_metrikler['shannon_entropisi']} bit")
    print(f"{'CLAHE (Adaptive)':<25} | {clahe_metrikler['dinamik_aralik']:<16} | {clahe_metrikler['rms_kontrast']:<14} | {clahe_metrikler['shannon_entropisi']} bit")
    print("-" * 74)

    baslik("AŞAMA 5: 2x3 Histogram & CDF Raporunun Kaydedilmesi")
    gorsel_haritasi = {
        "1. Düşük Kontrastlı Orijinal": ham_gorsel,
        "2. Global Histogram Eşitleme": global_esitlenmis,
        "3. CLAHE İyileştirmesi": clahe_gorsel
    }
    cikti_yolu = proje_kok / "ciktilar" / "histogram_analiz_raporu.png"
    kaydedilen = HistogramGorsellestirici.analiz_raporu_ciz(gorsel_haritasi, cikti_yolu)

    print(f"[V] Karşılaştırma raporu kaydedildi: {kaydedilen.name}")
    print(f"[V] Tam Yol: {kaydedilen}")
    print("\n[V] Day 9: Görüntü Histogramı Analizörü ve Kontrast İyileştirme tamamlandı.")


if __name__ == "__main__":
    main()
