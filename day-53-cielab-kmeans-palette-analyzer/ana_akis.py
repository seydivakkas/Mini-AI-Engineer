"""
Day 53: CIELAB Renk Uzayında K-Means & Delta-E 2000 Hassas Tolerans Analizi Ana Yürütme Betiği.
"""

import os
import numpy as np
from src.renk_uzayi_donusturucu import RenkUzayiDonusturucu
from src.cielab_kmeans_analizor import CIELABKMeansPaletAnalizoru
from src.delta_e_hesaplayici import DeltaEHesaplayici
from src.gorsellestirici import PaletAnalizGorsellestirici


def sentetik_tekstil_paleti_uret(sapma_miktari: float = 0.0) -> np.ndarray:
    """Gerçekçi 4 renkli geometrik jakarlı tekstil kumaşı ve kontrollü boyama lot sapması üretir."""
    np.random.seed(42)
    h, w = 360, 360
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 4 Temel Referans Rengi (sRGB):
    # 1. Gece Mavisi: [32, 54, 92]
    # 2. Pişmiş Toprak (Terracotta): [192, 75, 48]
    # 3. Sıcak Bej: [228, 215, 185]
    # 4. Orman Yeşili: [45, 98, 64]
    ana_renkler = np.array([
        [32, 54, 92],
        [192, 75, 48],
        [228, 215, 185],
        [45, 98, 64]
    ], dtype=float)

    # Numune partisinde boyama lot sapması (Dye lot shift)
    if sapma_miktari > 0:
        sapma_vektoru = np.array([
            [sapma_miktari * 4.0, -sapma_miktari * 2.0, sapma_miktari * 3.0],
            [sapma_miktari * 3.0, sapma_miktari * 4.0, -sapma_miktari * 2.0],
            [-sapma_miktari * 2.0, sapma_miktari * 3.0, sapma_miktari * 2.0],
            [sapma_miktari * 2.0, -sapma_miktari * 3.0, sapma_miktari * 4.0]
        ])
        ana_renkler = np.clip(ana_renkler + sapma_vektoru, 0, 255)

    renkler_uint8 = ana_renkler.astype(np.uint8)

    # Geometrik jakar deseni
    for i in range(h):
        for j in range(w):
            bolge = ((i // 60) + (j // 60)) % 4
            gurultu = np.random.normal(0, 3, 3)
            piksel = np.clip(renkler_uint8[bolge] + gurultu, 0, 255).astype(np.uint8)
            img[i, j] = piksel

    return img


def main():
    print("=" * 85)
    print(">>> DAY 53: CIELAB RENK UZAYINDA K-MEANS & DELTA-E 2000 HASSAS TOLERANS ANALİZİ")
    print("=" * 85)

    # 1. Referans Standart ve Üretim Partisi Kumaş Görsellerinin Üretimi
    print("\n[+] 1. Adım: Master Referans Standart ve Numune Kumaş Görselleri Hazırlanıyor...")
    hedef_gorsel = sentetik_tekstil_paleti_uret(sapma_miktari=0.0)
    numune_gorsel = sentetik_tekstil_paleti_uret(sapma_miktari=2.5)  # 2.5 birim kontrollü lot sapması
    print(f"    - Referans Görsel Boyutu: {hedef_gorsel.shape[1]}x{hedef_gorsel.shape[0]} px")
    print(f"    - Numune Görsel Boyutu  : {numune_gorsel.shape[1]}x{numune_gorsel.shape[0]} px")

    # 2. CIELAB Uzayında K-Means Dominant Palet Çıkarımı
    print("\n[+] 2. Adım: CIELAB (L*a*b*) Uzayında K-Means Kümeleme (K=4)...")
    analizor = CIELABKMeansPaletAnalizoru(k_renk=4, random_state=42)
    hedef_analiz = analizor.palet_cikar(hedef_gorsel)
    numune_analiz = analizor.palet_cikar(numune_gorsel)

    print("\n    >>> REFERANS STANDART DOMİNANT PALET:")
    for p in hedef_analiz["palet"]:
        print(f"      * [Sıra #{p['sira']}] HEX: {p['hex']} | L*:{p['lab'][0]:>5.1f} a*:{p['lab'][1]:>5.1f} b*:{p['lab'][2]:>5.1f} | Oran: %{p['yuzde']:>4.1f}")

    # 3. Eşleşen Renk Çiftleri Arasında Delta-E 76 ve CIEDE2000 Hesabı
    print("\n[+] 3. Adım: Delta-E 1976 ve CIEDE2000 Algısal Renk Sapması Analizi...")
    karsilastirma = []
    toplam_de00 = 0.0

    for idx in range(4):
        lab_ref = hedef_analiz["palet"][idx]["lab"]
        lab_num = numune_analiz["palet"][idx]["lab"]

        de76 = DeltaEHesaplayici.delta_e_76(lab_ref, lab_num)
        de00 = DeltaEHesaplayici.delta_e_2000(lab_ref, lab_num)
        tolerans = DeltaEHesaplayici.tolerans_degerlendir(de00)

        toplam_de00 += de00
        karsilastirma.append({
            "sira": idx + 1,
            "ref_hex": hedef_analiz["palet"][idx]["hex"],
            "num_hex": numune_analiz["palet"][idx]["hex"],
            "delta_e_76": de76,
            "delta_e_2000": de00,
            "tolerans": tolerans
        })

        print(f"      * Renk #{idx+1} ({hedef_analiz['palet'][idx]['hex']} -> {numune_analiz['palet'][idx]['hex']}): "
              f"dE76={de76:>5.2f} | dE00={de00:>5.2f} | Durum: {tolerans['seviye']} ({tolerans['kod']})")

    ortalama_de00 = float(round(toplam_de00 / 4.0, 2))
    genel_tolerans = DeltaEHesaplayici.tolerans_degerlendir(ortalama_de00)

    print(f"\n    >>> PARTİ GENELİ ORTALAMA CIEDE2000: {ortalama_de00:.2f} dE00")
    print(f"    >>> GENEL ENDÜSTRİYEL KARAR        : {genel_tolerans['seviye']} ({genel_tolerans['aciklama']})")

    # 4. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85)
    print(">>> 4. 6 PANELLİ KOLORİMETRİ VE DELTA-E 2000 TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_yolu = PaletAnalizGorsellestirici.panel_ciz(
        hedef_gorsel=hedef_gorsel,
        numune_gorsel=numune_gorsel,
        hedef_analiz=hedef_analiz,
        numune_analiz=numune_analiz,
        karsilastirma_sonuclari=karsilastirma,
        ortalama_de00=ortalama_de00,
        tolerans_ozet=genel_tolerans,
        hedef_path="day-53-cielab-kmeans-palette-analyzer/ciktilar/cielab_palet_tolerans_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}")
    print("=" * 85)
    print("DAY 53: CIELAB K-MEANS VE DELTA-E 2000 PALET ANALİZÖRÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
