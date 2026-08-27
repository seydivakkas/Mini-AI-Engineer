"""Günün Ana Çalıştırma Akışı: Vektör Benzerliği Tabanlı Görsel Arama Motoru.

Bu betik; 4 farklı ürün kategorisinde (Seramik Vazo, Mavi Kumaş, Altın Yıldız, Ahşap Parke)
toplam 16 katalog görseli üretir, hibrit öznitelik vektörlerini çıkararak indeksler,
yeni bir sorgu görseli için k-NN (Cosine ve L2) benzerlik aramasını koşturur ve
sonuçları karşılaştırmalı görsel rapor olarak kaydeder.
"""

import sys
import time
from pathlib import Path
from typing import Dict

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.vektor_cikarici import GorselVektorCikarici
from src.knn_arama_motoru import GorselAramaMotoru
from src.gorsellestirici import AramaGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 78
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def sentetik_katalog_uret() -> Dict[str, np.ndarray]:
    """4 kategoride 16 adet 128x128 BGR sentetik katalog görseli üretir."""
    katalog = {}
    h, w = 128, 128

    # --- Kategori 1: Seramik Vazo (Terracotta / Kiremit Kırmızısı) ---
    for varyant, bgr, genislik in [
        ("vazo_kirmizi_klasik", (35, 60, 210), 30),
        ("vazo_terracotta_ince", (45, 80, 200), 22),
        ("vazo_koyu_seramik", (30, 45, 170), 34),
        ("vazo_altin_bantli", (35, 65, 205), 28)
    ]:
        img = np.full((h, w, 3), (225, 225, 230), dtype=np.uint8)  # Açık gri fon
        cv2.ellipse(img, (64, 75), (genislik, 40), 0, 0, 360, bgr, -1)
        cv2.rectangle(img, (64 - int(genislik * 0.5), 25), (64 + int(genislik * 0.5), 45), bgr, -1)
        if "altin" in varyant:
            cv2.ellipse(img, (64, 75), (genislik, 8), 0, 0, 360, (20, 200, 240), -1)
        katalog[varyant] = img

    # --- Kategori 2: Mavi Dokulu Kumaş (Lacivert / Çizgili / Dokuma) ---
    for varyant, ton, adim in [
        ("kumas_mavi_cizgili", (200, 120, 30), 8),
        ("kumas_lacivert_dokuma", (160, 70, 20), 5),
        ("kumas_petrol_desenli", (170, 140, 40), 12),
        ("kumas_buz_mavisi", (225, 190, 90), 10)
    ]:
        img = np.full((h, w, 3), ton, dtype=np.uint8)
        for i in range(0, w, adim):
            cv2.line(img, (i, 0), (i + 40, h), (ton[0] - 30, ton[1] - 30, ton[2] - 10), 2)
        katalog[varyant] = img

    # --- Kategori 3: Altın Yıldız / Geometrik Rozet ---
    for varyant, bgr in [
        ("yildiz_altin_sekizgen", (30, 210, 245)),
        ("yildiz_sari_parlak", (40, 230, 255)),
        ("yildiz_bronz_geometrik", (35, 160, 200)),
        ("yildiz_kehribar_rozet", (20, 180, 225))
    ]:
        img = np.full((h, w, 3), (35, 35, 40), dtype=np.uint8)  # Koyu antrasit fon
        pts = np.array([
            [64, 20], [75, 48], [105, 48], [80, 68],
            [90, 98], [64, 78], [38, 98], [48, 68],
            [23, 48], [53, 48]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], bgr)
        katalog[varyant] = img

    # --- Kategori 4: Ahşap Parke (Kahverengi Yatay Dokular) ---
    for varyant, ana_renk in [
        ("ahsap_mese_parke", (60, 110, 160)),
        ("ahsap_ceviz_kaplama", (40, 80, 130)),
        ("ahsap_koyu_kereste", (30, 60, 100)),
        ("ahsap_rustik_tahta", (70, 120, 175))
    ]:
        img = np.full((h, w, 3), ana_renk, dtype=np.uint8)
        for y in range(0, h, 14):
            cv2.line(img, (0, y), (w, y), (ana_renk[0] - 25, ana_renk[1] - 25, ana_renk[2] - 25), 2)
            cv2.circle(img, (int(w * 0.3), y + 7), 3, (ana_renk[0] - 20, ana_renk[1] - 20, ana_renk[2] - 20), -1)
        katalog[varyant] = img

    return katalog


def sorgu_gorseli_uret() -> np.ndarray:
    """Katalogda birebir aynısı olmayan, hafif deforme edilmiş bir Seramik Vazo sorgusu üretir."""
    h, w = 128, 128
    img = np.full((h, w, 3), (230, 230, 235), dtype=np.uint8)
    bgr = (40, 65, 215)  # Kiremit kırmızısı
    # Hafif sağa kaydırılmış vazo
    cv2.ellipse(img, (67, 72), (29, 42), 5, 0, 360, bgr, -1)
    cv2.rectangle(img, (53, 23), (81, 44), bgr, -1)
    # Hafif gürültü ekle
    gurultu = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
    img_gurultulu = np.clip(img.astype(np.int16) + gurultu, 0, 255).astype(np.uint8)
    return img_gurultulu


def main() -> None:
    baslik("AŞAMA 1: Sentetik Katalog ve Çok-Kategorili Görsellerin Üretilmesi")
    katalog = sentetik_katalog_uret()
    print(f"[+] Katalogdaki Toplam Ürün Adedi: {len(katalog)} adet")
    kategoriler = set(k.split("_")[0] for k in katalog.keys())
    print(f"[+] Ürün Kategorileri            : {', '.join(sorted(kategoriler))}")

    baslik("AŞAMA 2: Çok Modaliteli Hibrit Vektörlerin İndekslenmesi")
    cikarici = GorselVektorCikarici(
        standart_boyut=(128, 128),
        agirlik_renk=1.2,
        agirlik_doku=0.8,
        agirlik_sekil=1.0
    )
    arama_motoru = GorselAramaMotoru(vektor_cikarici=cikarici, varsayilan_metrik="cosine")

    t0 = time.perf_counter()
    indekslenen_sayisi = arama_motoru.katalog_toplu_indeksle(katalog)
    sure_indeks = (time.perf_counter() - t0) * 1000.0

    vektor_boyutu = arama_motoru.katalog_matrisi.shape[1]
    print(f"[V] Başarıyla İndekslenen Görsel : {indekslenen_sayisi} adet")
    print(f"[V] Hibrit Vektör Boyutu (D)    : {vektor_boyutu} boyut (Renk 64 + LBP 10 + HOG 1568)")
    print(f"[V] Toplam İndeksleme Süresi    : {sure_indeks:.2f} ms ({sure_indeks / indekslenen_sayisi:.2f} ms/görsel)")

    baslik("AŞAMA 3: Görsel Sorgu ve k-NN (Kosinüs Benzerliği) ile Top-5 Arama")
    sorgu = sorgu_gorseli_uret()

    t1 = time.perf_counter()
    sonuclar_cosine = arama_motoru.en_yakin_k_ara(sorgu, k=5, metrik="cosine")
    sure_sorgu = (time.perf_counter() - t1) * 1000.0

    print(f"[+] Sorgu Tamamlanma Süresi     : {sure_sorgu:.3f} ms (Mikrosaniyeler mertebesinde!)")
    print("\n--- TOP-5 EŞLEŞME TABLOSU (Kosinüs Benzerliği) ---")
    print(f"{'Sıra':<6} | {'Katalog Etiketi':<25} | {'Cosine Mesafe':<14} | {'Benzerlik (%)':<14} | {'Kategori Eşleşti mi?'}")
    print("-" * 78)
    for r in sonuclar_cosine:
        eslesme = "EVET (Doğru)" if r.etiket.startswith("vazo") else "HAYIR"
        print(f"#{r.sira:<5} | {r.etiket:<25} | {r.mesafe:<14.4f} | %{r.benzerlik_yuzdesi:<13.1f} | {eslesme}")
    print("-" * 78)

    # L2 Metriği ile Karşılaştırma
    sonuclar_l2 = arama_motoru.en_yakin_k_ara(sorgu, k=5, metrik="l2")
    print(f"\n[+] L2 Metriği Kontrolü: 1. Sıradaki Eşleşme -> {sonuclar_l2[0].etiket} (L2 Mesafe: {sonuclar_l2[0].mesafe:.4f})")

    baslik("AŞAMA 4: Görsel Arama Rapor Çizelgesinin Kaydedilmesi")
    cikti_klasoru = proje_kok / "ciktilar"
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    rapor_dosyasi = cikti_klasoru / "gorsel_arama_raporu.png"

    kaydedilen = AramaGorsellestirici.arama_raporu_ciz(
        sorgu_gorseli_bgr=sorgu,
        sonuclar=sonuclar_cosine,
        kullanilan_metrik="Cosine",
        dosya_yolu=rapor_dosyasi
    )
    print(f"[V] Görsel arama raporu kaydedildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 17: Vektör Benzerliği Tabanlı Görsel Arama başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
