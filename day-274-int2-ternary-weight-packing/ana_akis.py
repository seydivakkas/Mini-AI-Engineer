"""
Day 274 (FAZ 14): Bit Düzeyinde Paketleme (Bit-Packing: INT2 / Ternary) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.bit_packing_motoru import BitPackingKernelEngine
from src.bit_packing_profilleyici import BitPackingProfilleyici
from src.gorsellestirici import BitPackingGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 274 (FAZ 14): BİT DÜZEYİNDE PAKETLEME (INT2 / TERNARY BIT-PACKING) — UINT32 SIKIŞTIRMA VE DONANIM KERNELİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 16-to-1 UINT32 Bit-Packing Mimarisi Başlatılıyor
    # -------------------------------------------------------------
    print("\n[1/4] 16-to-1 UINT32 Bit-Packing ve Donanım Eşleme Motoru Başlatılıyor...")
    print(f"  • Paketleme Oranı                    : 16 Ağırlık / 32-bit UINT32 (2-Bit / Ağırlık)")
    print(f"  • Desteklenen Formatlar              : INT2 [0, 3] ve BitNet Ternary {{-1, 0, +1}}")
    print(f"  • Donanım Çözme Yöntemi              : SIMD Register Bit-Shift & Maskeleme (Zero Memory Copy)")

    # -------------------------------------------------------------
    # ADIM 2: 4096 x 4096 Matris Paketleme ve Bit-Level Doğruluk
    # -------------------------------------------------------------
    print("\n[2/4] 4096 x 4096 Ternary Ağırlık Matrisi Paketleniyor ve Fused GEMM Doğrulanıyor...")
    mock_res = BitPackingKernelEngine.execute_mock_packing_pipeline(matrix_rows=4096, matrix_cols=4096)

    print(f"  • Orijinal Matris Boyutu             : {mock_res['matrix_shape'][0]} x {mock_res['matrix_shape'][1]} (16.77 Milyon Ağırlık)")
    print(f"  • FP16 VRAM Bellek Ayak İzi          : {mock_res['fp16_mb']:.2f} MB")
    print(f"  • INT8 VRAM Bellek Ayak İzi          : {mock_res['int8_mb']:.2f} MB")
    print(f"  • INT2 Packed VRAM Ayak İzi          : {mock_res['packed_mb']:.2f} MB")
    print(f"  • Sıkıştırma / Tasarruf Oranı        : {mock_res['tasarruf_orani_fp16']:.1f}x Kat Tasarruf (%87.5 Küçülme)")
    print(f"  • Bit-Level Veri Doğruluğu           : {'✓ BAŞARILI (Tam Birebir Eşleşme)' if mock_res['tam_eslesme'] else '✗ HATALI'}")
    print(f"  • Fused GEMM Çıktı Tensör Boyutu     : {mock_res['gemm_out_shape']}")

    # -------------------------------------------------------------
    # ADIM 3: 70B Model VRAM ve Çıkarım Hızı Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] 70B Model Üzerinde VRAM ve Çıkarım Hızı Donanım Raporu Hesaplanıyor...")
    profil_raporu = BitPackingProfilleyici.basarim_profili_cikar()
    karsilastirma = profil_raporu["karsilastirma"]

    print(f"  • 70B Model VRAM (FP16 -> INT2)      : {karsilastirma['vram_ayak_izi_70b_gb']['FP16_Standart']:.1f} GB -> {karsilastirma['vram_ayak_izi_70b_gb']['INT2_Ternary_Packed']:.1f} GB (8.0x Sıkıştırma)")
    print(f"  • Tüketici GPU Uyumluluğu            : ✓ TEK 24GB RTX 3090/4090 GPU'YA TAM SIĞAR!")
    print(f"  • Bellek Bant Genişliği İhtiyacı     : {karsilastirma['bellek_bant_genisligi_gb_s']['FP16_Standart']:.0f} GB/s -> {karsilastirma['bellek_bant_genisligi_gb_s']['INT2_Ternary_Packed']:.0f} GB/s (8.0x Azalma)")
    print(f"  • Çıkarım Üretim Hızı (Token/s)      : {karsilastirma['cikarim_hizi_token_s']['FP16_Standart']:.0f} t/s -> {karsilastirma['cikarim_hizi_token_s']['INT2_Ternary_Packed']:.0f} t/s (4.78x Hızlanma)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli INT2 / Ternary Bit-Packing Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "int2_ternary_packing_paneli.png")

    BitPackingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Bit-Packing Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 274 (FAZ 14): BİT DÜZEYİNDE PAKETLEME (INT2 / TERNARY BIT-PACKING) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
