"""
Day 48: K-Means ile Denetimsiz Görüntü & Özellik Bölütleme Ana Yürütme Betiği.
"""

import os
import numpy as np
from src.kmeans_bolutleyici import KMeansGorselBolutleyici
from src.kume_optimizasyonu import KumeOptimizatoru
from src.gorsellestirici import KMeansBolutlemeGorsellestirici


def sentetik_hali_deseni_uret(H: int = 240, W: int = 240) -> np.ndarray:
    """Belirgin renk ve doku bölgelerine sahip sentetik test görüntüsü üretir."""
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # 1. Zemin: Derin Lacivert (#1B263B)
    img[:, :] = [27, 38, 59]

    # 2. Dış Çerçeve: Bordo (#780000)
    img[20:H-20, 20:W-20] = [120, 0, 0]

    # 3. İç Alan: Altın Sarısı (#D4A373)
    img[50:H-50, 50:W-50] = [212, 163, 115]

    # 4. Merkez Madalyon: Zümrüt Yeşili (#2D6A4F)
    y, x = np.ogrid[:H, :W]
    merkez_mask = ((x - W // 2) ** 2 + (y - H // 2) ** 2) <= (45 ** 2)
    img[merkez_mask] = [45, 106, 79]

    # Doğal gürültü ekleme
    gurultu = np.random.normal(0, 8, img.shape).astype(np.int16)
    img_gurultulu = np.clip(img.astype(np.int16) + gurultu, 0, 255).astype(np.uint8)

    return img_gurultulu


def main():
    print("=" * 85)
    print(">>> DAY 48: K-MEANS İLE DENETİMSİZ GÖRÜNTÜ & ÖZELLİK BÖLÜTLEME")
    print("=" * 85)

    # 1. Sentetik Test Görüntüsünün Üretilmesi
    H, W = 240, 240
    gorsel = sentetik_hali_deseni_uret(H, W)
    print(f"[+] Test Görüntüsü Oluşturuldu: {W}x{H} Piksel, 3 Kanal (RGB)")

    # 2. Optimal Küme Sayısının (K*) Belirlenmesi (Elbow & Silhouette)
    print("\n[+] 1. Adım: Elbow ve Silhouette Analizi ile Optimal K Araştırması...")
    duz_pikseller = (gorsel.reshape(-1, 3).astype(np.float32)) / 255.0
    kume_analizi = KumeOptimizatoru.en_iyi_k_bul(duz_pikseller, k_araligi=(2, 7))

    print(f"    - Taranan K Değerleri      : {kume_analizi['k_degerleri']}")
    print(f"    - Silhouette Skorları      : {kume_analizi['silhouette_degerleri']}")
    print(f"    - Optimal Küme Sayısı (K*) : {kume_analizi['en_iyi_k']} (Maksimum Silhouette: {kume_analizi['en_iyi_silhouette']})")

    # 3. K-Means ile Renk Kuantalama (RGB Only)
    en_iyi_k = kume_analizi["en_iyi_k"]
    print(f"\n[+] 2. Adım: K={en_iyi_k} ile Renk Kuantalama (RGB Uzayı)...")
    bolutleyici = KMeansGorselBolutleyici(k_kume=en_iyi_k, uzamsal_agirlik=0.40)
    kuantalanmis_gorsel, maske_rgb, merkezler = bolutleyici.renk_kuantalama_uygula(gorsel)
    print(f"    - Renk Kuantalama Tamamlandı: {len(merkezler)} Hakim Merkez Renk Belirlendi")

    # 4. K-Means ile Uzamsal + Renk Bölütleme (RGB + XY)
    print(f"\n[+] 3. Adım: [R, G, B, alpha*X, alpha*Y] Füzyonu ile Uzamsal Bölütleme...")
    uzamsal_sonuc = bolutleyici.uzamsal_bolutleme_uygula(gorsel)

    print("    - Bölütlenmiş Küme Alan Dağılımı:")
    for k, yuzde in uzamsal_sonuc["alan_yuzdeleri"].items():
        rgb = [int(c * 255) for c in uzamsal_sonuc["kume_renkleri"][k]]
        print(f"      • Küme {k+1}: %{yuzde:5.2f} Alan | Ortalama RGB: {rgb}")

    # 5. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85)
    print(">>> 4. 6 PANELLİ K-MEANS BÖLÜTLEME TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_yolu = KMeansBolutlemeGorsellestirici.panel_ciz(
        orijinal_gorsel=gorsel,
        kuantalanmis_gorsel=kuantalanmis_gorsel,
        uzamsal_sonuc=uzamsal_sonuc,
        kume_analizi=kume_analizi,
        hedef_path="day-48-kmeans-unsupervised-segmentation/ciktilar/kmeans_bolutleme_paneli.png"
    )
    print(f"[+] 6 Panelli Bölütleme Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}")
    print("=" * 85)
    print("DAY 48: K-MEANS DENETİMSİZ BÖLÜTLEME PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
