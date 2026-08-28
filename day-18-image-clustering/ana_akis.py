"""Day 18 Ana Çalıştırma Akışı: Etiketsiz Görsellerin Otomatik Kümelenmesi.

Bu betik; 4 farklı görsel kategorisinde (Seramik Vazo, Mavi Kumaş, Altın Rozet, Ahşap Parke)
ve 2 adet sentetik gürültü/aykırı görsel içeren toplam 18 etiketsiz görsel üretir.
Hibrit öznitelik vektörlerini çıkarır, K-Means (Optimal K Analizi), DBSCAN ve
Agglomerative kümeleme algoritmalarını koşturur, kümeleme kalitelerini (Silhouette,
Davies-Bouldin, Calinski-Harabasz) kıyaslar ve görsel raporu kaydeder.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini Python path'e ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.vektor_cikarici import GorselVektorCikarici
from src.kumeleme_motoru import GorselKumelemeMotoru, KumelemeSonucu
from src.gorsellestirici import KumeGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def sentetik_veri_seti_uret() -> Tuple[List[str], List[np.ndarray]]:
    """4 kategoride 16 düzenli ve 2 adet aykırı/gürültülü görsel üretir."""
    etiketler = []
    gorseller = []
    h, w = 128, 128

    # --- Kategori 1: Seramik Vazo (Terracotta / Kırmızı Tonları) ---
    for varyant, bgr, genislik in [
        ("vazo_kirmizi_1", (35, 60, 210), 30),
        ("vazo_terracotta_2", (45, 80, 200), 22),
        ("vazo_koyu_seramik_3", (30, 45, 170), 34),
        ("vazo_altin_bantli_4", (35, 65, 205), 28),
    ]:
        img = np.full((h, w, 3), (225, 225, 230), dtype=np.uint8)
        cv2.ellipse(img, (64, 75), (genislik, 40), 0, 0, 360, bgr, -1)
        cv2.rectangle(
            img,
            (64 - int(genislik * 0.5), 25),
            (64 + int(genislik * 0.5), 45),
            bgr,
            -1,
        )
        if "altin" in varyant:
            cv2.ellipse(img, (64, 75), (genislik, 8), 0, 0, 360, (20, 200, 240), -1)
        etiketler.append(varyant)
        gorseller.append(img)

    # --- Kategori 2: Mavi Çizgili Kumaş (Lacivert / Petrol / Gökyüzü) ---
    for varyant, ton, adim in [
        ("kumas_mavi_cizgili_1", (200, 120, 30), 8),
        ("kumas_lacivert_dokuma_2", (160, 70, 20), 5),
        ("kumas_petrol_desenli_3", (170, 140, 40), 12),
        ("kumas_buz_mavisi_4", (225, 190, 90), 10),
    ]:
        img = np.full((h, w, 3), ton, dtype=np.uint8)
        for i in range(0, w, adim):
            cv2.line(
                img,
                (i, 0),
                (i + 40, h),
                (max(0, ton[0] - 30), max(0, ton[1] - 30), max(0, ton[2] - 10)),
                2,
            )
        etiketler.append(varyant)
        gorseller.append(img)

    # --- Kategori 3: Altın Rozet / Yıldız (Sarı / Parlak Geometrik) ---
    for varyant, boyut in [
        ("rozet_altin_buyuk_1", 38),
        ("rozet_altin_orta_2", 28),
        ("rozet_altin_cift_halka_3", 32),
        ("rozet_altin_parlak_4", 25),
    ]:
        img = np.full((h, w, 3), (35, 35, 40), dtype=np.uint8)
        cv2.circle(img, (64, 64), boyut, (30, 215, 255), -1)
        cv2.circle(img, (64, 64), int(boyut * 0.7), (20, 170, 230), -1)
        cv2.circle(img, (64, 64), int(boyut * 0.35), (255, 255, 255), -1)
        etiketler.append(varyant)
        gorseller.append(img)

    # --- Kategori 4: Ahşap Parke (Kahverengi / Yatay Doku) ---
    for varyant, ton in [
        ("ahsap_mese_1", (50, 90, 140)),
        ("ahsap_ceviz_2", (30, 60, 100)),
        ("ahsap_cam_3", (70, 120, 180)),
        ("ahsap_antika_4", (25, 50, 85)),
    ]:
        img = np.full((h, w, 3), ton, dtype=np.uint8)
        for y in range(0, h, 6):
            cizgi_renk = (
                max(0, ton[0] - 15),
                max(0, ton[1] - 15),
                max(0, ton[2] - 15),
            )
            cv2.line(img, (0, y), (w, y), cizgi_renk, 1)
        etiketler.append(varyant)
        gorseller.append(img)

    # --- Aykırı / Gürültülü Görseller (Outliers for DBSCAN) ---
    # Gürültü 1: Saf Beyaz Üzerinde Rastgele Renkli Pikseller
    gurultu_1 = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    etiketler.append("aykiri_rastgele_gurultu")
    gorseller.append(gurultu_1)

    # Gürültü 2: Neon Yeşil ve Pembe Kare Izgara
    gurultu_2 = np.zeros((h, w, 3), dtype=np.uint8)
    gurultu_2[::16, :] = (0, 255, 0)
    gurultu_2[:, ::16] = (255, 0, 255)
    etiketler.append("aykiri_neon_izgara")
    gorseller.append(gurultu_2)

    return etiketler, gorseller


def main() -> None:
    """Day 18 görsel kümeleme tam akışını çalıştırır."""
    baslik("AŞAMA 1: Sentetik Görsel Veri Setinin Hazırlanması")
    etiketler, gorseller = sentetik_veri_seti_uret()
    print(f"[+] Toplam {len(gorseller)} adet görsel üretildi:")
    print("    - 4x Vazo (Kırmızı/Terracotta)")
    print("    - 4x Kumaş (Mavi Çizgili/Dokulu)")
    print("    - 4x Rozet (Altın Parlak Geometrik)")
    print("    - 4x Ahşap (Kahverengi Doğal Doku)")
    print("    - 2x Aykırı / Gürültü Görseli (Outliers)")

    baslik("AŞAMA 2: Çok Modaliteli Hibrit Öznitelik Çıkarımı")
    cikarici = GorselVektorCikarici(hedef_boyut=(64, 64))
    t0 = time.perf_counter()
    vektorler = np.array([cikarici.cikar(img) for img in gorseller])
    t_cikarim = (time.perf_counter() - t0) * 1000

    print(f"[+] Öznitelik Çıkarımı Tamamlandı: {t_cikarim:.2f} ms")
    print(f"[+] Embedding Matrisi Boyutu: {vektorler.shape} (N={vektorler.shape[0]}, D={vektorler.shape[1]})")
    print(f"[+] L2 Norm Doğrulaması: {[round(float(np.linalg.norm(v)), 4) for v in vektorler[:3]]} (Birim Küre)")

    baslik("AŞAMA 3: K-Means ile Optimal K Taraması ve Silhouette Analizi")
    motor = GorselKumelemeMotoru(random_state=42)
    en_iyi_k, k_skorlari, en_iyi_kmeans = motor.en_iyi_k_bul_kmeans(
        vektorler, k_araligi=range(2, 7)
    )

    print("K Değeri | Silhouette Skoru | Değerlendirme")
    print("-" * 50)
    for k, skor in k_skorlari.items():
        isaret = " <-- EN İYİ (ZİRVE)" if k == en_iyi_k else ""
        print(f"  K={k:<4} | {skor:<16.4f} | {isaret}")
    print(f"\n[+] Seçilen Optimal Küme Sayısı: K={en_iyi_k}")
    print(f"[+] Optimal K-Means Özeti: {en_iyi_kmeans.ozet()}")

    baslik("AŞAMA 4: DBSCAN Yoğunluk Tabanlı Kümeleme ve Anomali Tespiti")
    sonuc_dbscan = motor.dbscan_kumele(
        vektorler, eps=0.35, min_samples=2, metric="cosine"
    )
    print(f"[+] DBSCAN Özeti: {sonuc_dbscan.ozet()}")
    print(f"    - Keşfedilen Küme Sayısı: {sonuc_dbscan.kume_sayisi}")
    print(f"    - Aykırı Değer / Gürültü Sayısı: {sonuc_dbscan.gurultu_sayisi}")
    
    # Gürültü olarak işaretlenen görselleri listele
    gurultu_indeksleri = np.where(sonuc_dbscan.etiketler == -1)[0]
    print(f"    - Aykırı Görsel Etiketleri: {[etiketler[i] for i in gurultu_indeksleri]}")

    baslik("AŞAMA 5: Hiyerarşik (Agglomerative) Kümeleme")
    sonuc_agg = motor.agglomerative_kumele(
        vektorler, n_clusters=4, metric="cosine", linkage="average"
    )
    print(f"[+] Agglomerative Özeti: {sonuc_agg.ozet()}")

    baslik("AŞAMA 6: Algoritma Kıyaslama ve Kümeleme Kalite Tablosu")
    print(f"{'Algoritma':<25} | {'Küme':<6} | {'Gürültü':<8} | {'Silhouette':<12} | {'Davies-Bouldin':<15} | {'Calinski-Harabasz'}")
    print("-" * 95)
    for res in [en_iyi_kmeans, sonuc_dbscan, sonuc_agg]:
        sil = f"{res.silhouette:.4f}" if res.silhouette is not None else "N/A"
        db = f"{res.davies_bouldin:.4f}" if res.davies_bouldin is not None else "N/A"
        ch = f"{res.calinski_harabasz:.2f}" if res.calinski_harabasz is not None else "N/A"
        print(f"{res.algoritma:<25} | {res.kume_sayisi:<6} | {res.gurultu_sayisi:<8} | {sil:<12} | {db:<15} | {ch}")
    print("-" * 95)
    print("(*) Silhouette: 1'e ne kadar yakınsa o kadar iyi ayrışma.")
    print("(*) Davies-Bouldin: 0'a ne kadar yakınsa o kadar kompakt ve ayrık kümeler.")
    print("(*) Calinski-Harabasz: Ne kadar yüksekse o kadar yoğun kümeler.")

    baslik("AŞAMA 7: Görselleştirme ve Analiz Raporunun Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    rapor_dosyasi = cikti_klasor / "kumeleme_raporu.png"

    KumeGorsellestirici.kumeleme_raporu_olustur(
        X=vektorler,
        gorseller=gorseller,
        kumeleme_sonucu=en_iyi_kmeans,
        k_skorlari=k_skorlari,
        hedef_dosya=rapor_dosyasi,
    )
    print(f"[+] Kümeleme teşhis raporu başarıyla üretildi: {rapor_dosyasi}")


if __name__ == "__main__":
    main()
