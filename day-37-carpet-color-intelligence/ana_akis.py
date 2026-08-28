"""
Day 37: Halı/Tekstil Renk Ayrıştırma & İplik Renk Oranları Çıkarımı Ana Yürütme Betiği.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from src.iplik_kumeleyici import IplikRenkKumeleyici
from src.katalog_esleyici import IplikKatalogEsleyici
from src.gorsellestirici import HaliRenkGorsellestirici


def sentetik_hali_deseni_olustur(genislik: int = 400, yukseklik: int = 300) -> Image.Image:
    """Geleneksel motifli sentetik çok renkli halı görseli üretir."""
    img = Image.new("RGB", (genislik, yukseklik), color=(228, 217, 198))  # Krem zemin
    draw = ImageDraw.Draw(img)

    # 1. Dış Çerçeve Bordür (Kraliyet Bordosu)
    draw.rectangle([10, 10, genislik - 10, yukseklik - 10], outline=(138, 28, 48), width=22)

    # 2. İç Madalyon Çerçevesi (Gece Mavisi)
    draw.rectangle([45, 45, genislik - 45, yukseklik - 45], outline=(24, 43, 73), width=12)

    # 3. Merkez Madalyon (Zümrüt Yeşili & Bordo)
    cx, cy = genislik // 2, yukseklik // 2
    draw.ellipse([cx - 80, cy - 60, cx + 80, cy + 60], fill=(32, 98, 65), outline=(138, 28, 48), width=6)

    # 4. Köşe Motifleri & Hardal Detayları
    draw.polygon([(45, 45), (100, 45), (45, 100)], fill=(204, 154, 45))
    draw.polygon([(genislik-45, 45), (genislik-100, 45), (genislik-45, 100)], fill=(204, 154, 45))
    draw.polygon([(45, yukseklik-45), (100, yukseklik-45), (45, yukseklik-100)], fill=(204, 154, 45))
    draw.polygon([(genislik-45, yukseklik-45), (genislik-100, yukseklik-45), (genislik-45, yukseklik-100)], fill=(204, 154, 45))

    # 5. Merkez Göbek Yıldızı (Krem & Hardal)
    draw.ellipse([cx - 30, cy - 25, cx + 30, cy + 25], fill=(204, 154, 45), outline=(228, 217, 198), width=3)

    # Dokuma pürüzlülüğü ekle (Noise)
    np_img = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, 3, np_img.shape).astype(np.int16)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(np_img)


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Halı Dokuma Deseninin ve Renk Uzayının Hazırlanması")
    print("=" * 80)

    hali_gorseli = sentetik_hali_deseni_olustur(400, 300)
    print(f"[+] Halı Görseli Üretildi: Boyut 400x300 Piksel (120,000 Piksel)")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: CIELAB Uzayında K-Means ile 5 Ana İplik Renginin Ayrıştırılması")
    print("=" * 80)

    kumeleyici = IplikRenkKumeleyici(k_iplik=5, max_iter=30, rastgele_durum=42)
    kumeleme_sonucu = kumeleyici.iplik_renklerini_ayristir(hali_gorseli)

    print(f"[+] Ayrıştırılan İplik Sayısı: {kumeleme_sonucu['iplik_sayisi']} adet")
    for iplik in kumeleme_sonucu["iplikler"]:
        print(f"    - {iplik['iplik_id']} | Sarfiyat: %{iplik['yuzde']:5.2f} | RGB: {iplik['rgb']} | HEX: {iplik['hex']} | LAB: {iplik['lab']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: CIE Delta-E 2000 ile Standart İplik Kataloğu Eşlemesi")
    print("=" * 80)

    esleyici = IplikKatalogEsleyici(tolerans_esigi=5.0)
    esleme_raporu = esleyici.esle_ve_raporla(kumeleme_sonucu["iplikler"])

    print(f"[+] Genel Parti Kalite Onayı: {'ONAYLANDI (PASS)' if esleme_raporu['genel_parti_onayi'] else 'REDDEDİLDİ (FAIL)'}")
    print(f"[+] Tolerans İçi İplik Sayısı: {esleme_raporu['tolerans_ici_iplik']}/{esleme_raporu['toplam_iplik_sayisi']}")
    print("\n[+] DETAYLI İPLİK KALİTE RAPORU:")
    print(f"{'İplik ID':<10} | {'Oran (%)':<8} | {'Katalog Adı':<22} | {'Katalog Kod':<10} | {'Delta-E 00':<10} | {'Kalite Kararı'}")
    print("-" * 88)
    for es in esleme_raporu["eslesmeler"]:
        print(f"{es['iplik_id']:<10} | %{es['iplik_yuzdesi']:<7.2f} | {es['katalog_ad']:<22} | {es['katalog_kod']:<10} | {es['delta_e_2000']:<10.2f} | {es['kalite_durumu']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: 6 Panelli Sektörel Halı Renk Zekası Panosunun Üretilmesi")
    print("=" * 80)

    cikis_resmi = HaliRenkGorsellestirici.hali_renk_paneli_ciz(
        orijinal_gorsel_rgb=np.array(hali_gorseli),
        kuantize_gorsel_rgb=kumeleme_sonucu["kuantize_gorsel_rgb"],
        kumeleme_sonucu=kumeleme_sonucu,
        esleme_raporu=esleme_raporu,
        hedef_path="day-37-carpet-color-intelligence/ciktilar/hali_renk_analiz_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 37: HALI/TEKSTİL RENK AYRIŞTIRMA & İPLİK ANALİZİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
