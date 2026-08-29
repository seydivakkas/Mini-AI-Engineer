"""
Day 164: Spatial Grounding ve Bounding Box Çıkarma (Spatial Grounding in VLMs) Ana Akışı.
RefCOCO Tarzı Doğal Dil Referanslama, [ymin, xmin, ymax, xmax] Koordinat Çıkarımı ve IoU Analizi.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.grounded_vlm_motoru import GroundedVLMMotoru
from src.gorsellestirici import SpatialGroundingGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 164 (FAZ 9): SPATIAL GROUNDING IN VLMS: [ymin, xmin, ymax, xmax] COORDINATE EXTRACTION & IoU")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. REFCOCO GROUNDING SENARYOLARININ DEĞERLENDİRİLMESİ
    # -------------------------------------------------------------
    print("\n[1/2] RefCOCO Doğal Dil Referans Senaryoları Değerlendiriliyor...")
    rapor = GroundedVLMMotoru.senaryolari_degerlendir()

    print("\n" + "-" * 105)
    print(f"{'Hedef Nesne':<28} | {'Tahmin Kutu':<22} | {'Gerçek Etiket (GT)':<22} | {'IoU Skoru':<10} | {'Durum'}")
    print("-" * 105)
    for s in rapor["senaryo_sonuclari"]:
        durum_str = "[BAŞARILI]" if s["dogru_mu"] else "[BAŞARISIZ]"
        print(
            f"{s['nesne_adi']:<28} | {str(s['tahmin_kutu']):<22} | "
            f"{str(s['gt_kutu']):<22} | %{s['iou']*100:<8.1f} | {durum_str}"
        )
    print("-" * 105)

    ozet = rapor["genel_ozet"]
    print("\nGENEL PERFORMANS METRİKLERİ:")
    print(f"  • Toplam Test Edilen Nesne : {ozet['toplam_nesne']}")
    print(f"  • Başarılı Tespit (IoU>=0.5): {ozet['dogru_tespit_sayisi']} / {ozet['toplam_nesne']}")
    print(f"  • RefCOCO mAP@0.5 Başarımı : %{ozet['map_50_yuzdesi']}")
    print(f"  • Ortalama IoU Skoru       : %{ozet['ortalama_iou']*100:.1f}")

    # -------------------------------------------------------------
    # 2. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[2/2] 6 Panelli Spatial Grounding Teşhis Panosu Üretiliyor...")
    gorsellestirici = SpatialGroundingGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "spatial_grounding_bounding_box_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 164: SPATIAL GROUNDING IN VLMs BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
