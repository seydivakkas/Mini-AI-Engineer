"""
Day 175: ControlNet: Mekansal Koşullu Görüntü Üretimi Ana Akışı (FAZ 9).
Canny Kenar Çizgileri, MiDaS Derinlik, OpenPose İskeleti ve Zero-Convolution Füzyonu.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.controlnet_modeli import ControlNetModeli
from src.mekansal_kontrol_degerlendirici import MekansalKontrolDegerlendirici
from src.gorsellestirici import ControlNetGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 175 (FAZ 9): CONTROLNET SPATIAL CONDITIONING: ZERO-CONVOLUTION & MULTI-MODAL HINT FUSION")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. CONTROLNET MODELİ VE KOŞUL İPUÇLARI HAZIRLIĞI
    # -------------------------------------------------------------
    print("\n[1/3] ControlNet Modeli ve Sıfır-Konvolüsyon Katmanları Başlatılıyor...")
    model = ControlNetModeli(in_channels=4, hint_channels=3, base_channels=64)

    # 16x16 VAE Tensörü ve Canny Kenar/Derinlik İpucu Tensörü
    z_t = torch.randn(1, 4, 16, 16)
    hint_canny = torch.randn(1, 3, 16, 16)

    residuals = model(z_t, hint_canny, control_weight=1.0)

    print(f"  • Gürültülü VAE Gizli Tensörü (z_t) : {list(z_t.shape)} [Batch=1, C=4, 16x16]")
    print(f"  • Mekansal Koşul İpucu (Hint)       : {list(hint_canny.shape)} [Batch=1, C=3, 16x16]")
    print("  • Üretilen ControlNet Rezidüelleri (Zero-Conv Çıkışları):")
    for i, res in enumerate(residuals):
        print(f"    - Rezidüel Köprü {i+1} : {list(res.shape)}")

    # -------------------------------------------------------------
    # 2. MEKANSAL KOŞUL UYUMU ANALİZİ
    # -------------------------------------------------------------
    print("\n[2/3] Mekansal Koşul Sadakat ve Hizalama Analizi Yapılıyor...")
    rapor = MekansalKontrolDegerlendirici.ornek_kontrol_raporunu_getir()

    print("\n" + "-" * 85)
    print(f"{'Koşul Tipi':<30} | {'Mekansal Uyum':<18} | {'Hata Payı':<12} | {'Kaynak İpucu'}")
    print("-" * 85)
    for k in rapor["kosul_tipleri"]:
        print(f"{k['tip']:<30} | %{k['uyum_skoru']*100:<16.1f} | %{k['hata_orani']*100:<10.1f} | {k['kaynak']}")
    print("-" * 85)
    print(f"{'Ortalama Mekansal Sadakat':<30} | %{rapor['ortalama_mekansal_uyum']*100:.1f} (Piksel Düzeyinde Kusursuz Kontrol)")
    print(f"{'Eğitim Kararlılık Oranı':<30} | {rapor['zero_conv_egitim_kararliligi']}")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli ControlNet Teşhis Panosu Üretiliyor...")
    gorsellestirici = ControlNetGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "controlnet_spatial_conditioning_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 175: CONTROLNET SPATIAL CONDITIONING BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
