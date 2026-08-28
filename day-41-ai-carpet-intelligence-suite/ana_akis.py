"""
Day 41: Uçtan Uca Çoklu Görev Halı Zekası Paketi (AI Carpet Intelligence Suite) Ana Yürütme Betiği.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from src.orkestrator import HaliZekasiOrkestrator
from src.gorsellestirici import HaliZekaPaketiGorsellestirici


def ornek_test_numunesi_uret(genislik: int = 400, yukseklik: int = 300) -> Image.Image:
    """Kontrollü kusurlu test halı numunesi üretir."""
    img = Image.new("RGB", (genislik, yukseklik), color=(228, 217, 198))  # Krem zemin
    draw = ImageDraw.Draw(img)

    # Klasik Hereke Deseni
    draw.rectangle([15, 15, genislik - 15, yukseklik - 15], outline=(138, 28, 48), width=18)
    draw.rectangle([45, 45, genislik - 45, yukseklik - 45], outline=(24, 43, 73), width=8)

    cx, cy = genislik // 2, yukseklik // 2
    draw.ellipse([cx - 70, cy - 50, cx + 70, cy + 50], fill=(32, 98, 65), outline=(204, 154, 45), width=4)

    # 1. Kusur: İplik Kopması (Warp run)
    draw.rectangle([70, 75, 220, 79], fill=(20, 20, 20))

    # 2. Kusur: Yağ Lekesi
    draw.ellipse([270, 170, 310, 210], fill=(30, 25, 20))

    arr = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def main():
    print("=" * 85)
    print(">>> DAY 41: BÜYÜK FİNAL - UÇTAN UCA ÇOKLU GÖREV HALI VE TEKSTİL ZEKASI PAKETİ")
    print("=" * 85)

    test_numunesi = ornek_test_numunesi_uret(400, 300)
    print("[+] Canlı Konveyör Bandı Kamera Girdisi Alındı (400x300 Piksel)")

    orkestrator = HaliZekasiOrkestrator()
    print("\n[+] 4 Yapay Zeka Motoru (Renk, Arama, Kusur, RAG) Paralel Olarak Çalıştırılıyor...")

    rapor = orkestrator.tam_denetim_yap(test_numunesi, k_iplik=4)

    print("\n" + "=" * 85)
    print(">>> 1. RENK ZEKASI & İPLİK SARFİYAT ANALİZİ:")
    print("=" * 85)
    for ip in rapor["renk_analizi"]["iplikler"]:
        print(f"    - {ip['iplik_id']} | Sarfiyat: %{ip['yuzde']:5.2f} | Eşleşen: {ip['katalog_ad']:<22} | Delta-E: {ip['delta_e_2000']:4.2f} ({ip['uyum_durumu']})")

    print("\n" + "=" * 85)
    print(">>> 2. GÖRSEL ARAMA & KATALOG EŞLEŞMESİ:")
    print("=" * 85)
    top_m = rapor["gorsel_arama"]["en_iyi_eslesme"]
    if top_m:
        print(f"    - En Yakın Ürün: [{top_m['id']}] {top_m['ad']}")
        print(f"    - Kategori     : {top_m['kategori']} | Görsel Benzerlik Skoru: %{top_m['benzerlik_skoru']:.2f}")

    print("\n" + "=" * 85)
    print(">>> 3. DOKUMA HATALARI & KUSUR TESPİTİ:")
    print("=" * 85)
    print(f"    - Toplam Kusur Sayısı: {rapor['kusur_tespiti']['kusur_sayisi']} Adet (Kritik: {rapor['kusur_tespiti']['kritik_kusur_sayisi']})")
    for k in rapor["kusur_tespiti"]["kusurlar"]:
        print(f"    - {k['kusur_id']} | Tür: {k['kusur_turu']:<18} | Şiddet: {k['siddet']:<12} | Alan: {k['alan']} px | Kutu: {k['kutu']}")

    print("\n" + "=" * 85)
    print(">>> 4. OTOMATİK SEKTÖREL RAG STANDART ÇÖZÜM REÇETELERİ:")
    print("=" * 85)
    for r in rapor["rag_cozum_onerileri"]:
        print(f"    [*] {r['kusur_turu']} ({r['siddet']}):")
        print(f"        Standart: {r['standart']}")
        print(f"        Reçete  : {r['oneri']}")

    print("\n" + "=" * 85)
    print(">>> 5. YÖNETİCİ FABRİKA KALİTE VE SEVKİYAT KARARI:")
    print("=" * 85)
    print(f"    - Genel Kalite Skoru  : %{rapor['genel_kalite_skoru']:.1f} / 100")
    print(f"    - Fabrika Hat Kararı  : {rapor['fabrika_karari']}")
    print(f"    - Sevkiyat Onayı      : {'ONAYLANDI (PASS)' if rapor['sevkiyat_onayi'] else 'REDDEDİLDİ (FAIL)'}")

    print("\n" + "=" * 85)
    print(">>> 6. KONSOLİDE 6 PANELLİ FABRİKA KONTROL PANELİNİN ÜRETİLMESİ")
    print("=" * 85)

    cikis_resmi = HaliZekaPaketiGorsellestirici.konsolide_panel_ciz(
        test_gorseli=test_numunesi,
        teftis_raporu=rapor,
        hedef_path="day-41-ai-carpet-intelligence-suite/ciktilar/hali_zeka_paketi_paneli.png"
    )
    print(f"[+] 6 Panelli Fabrika Kontrol Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 85)
    print("DAY 41 & FAZ 2B: SEKTÖREL HALI VE TEKSTİL ZEKASI BÜYÜK FİNALİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
