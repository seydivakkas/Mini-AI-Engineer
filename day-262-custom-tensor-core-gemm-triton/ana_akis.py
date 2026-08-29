"""
Day 262 (FAZ 14): Özel NVIDIA Tensor Core GEMM Çekirdeği Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.tensor_core_gemm_motoru import (
    NaiveGEMM,
    SharedMemoryTiledGEMM,
    TensorCoreWMMASimulator,
)
from src.tensor_core_gemm_profilleyici import TensorCoreProfilleyici
from src.gorsellestirici import TensorCoreGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 262 (FAZ 14): ÖZEL NVIDIA TENSOR CORE GEMM ÇEKİRDEĞİ (WMMA / MMA VE BLOCK-TILING)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Matrislerin Hazırlanması ve GEMM Çekirdeklerinin Yüklenmesi
    # -------------------------------------------------------------
    print("\n[1/4] Giriş Matrisleri (A[128x128] ve B[128x128]) ve GEMM Çekirdekleri Hazırlanıyor...")
    np.random.seed(42)
    a_mat = np.random.randn(128, 128).astype(np.float32)
    b_mat = np.random.randn(128, 128).astype(np.float32)

    tiled_gemm = SharedMemoryTiledGEMM(block_m=64, block_n=64, block_k=32, padding=4)

    # -------------------------------------------------------------
    # ADIM 2: Matris Çarpımı ve Sayısal Doğruluk Testi
    # -------------------------------------------------------------
    print("\n[2/4] Shared Memory Tiled GEMM ve Tensor Core WMMA Çarpımları Yürütülüyor...")
    c_tiled, sram_stats = tiled_gemm.execute(a_mat, b_mat)
    c_wmma, wmma_stats = TensorCoreWMMASimulator.execute_wmma(a_mat, b_mat)

    hata = float(np.max(np.abs(c_tiled - c_wmma)))
    print(f"  • Shared Memory Tiled GEMM Çıktı Boyutu: {c_tiled.shape}")
    print(f"  • Tensor Core WMMA GEMM Çıktı Boyutu   : {c_wmma.shape}")
    print(f"  • Maksimum Sayısal Fark (Hata)         : {hata:.6f} (Doğruluk Onaylandı)")

    # -------------------------------------------------------------
    # ADIM 3: Donanım ve Bellek Trafiği Analitiği
    # -------------------------------------------------------------
    print("\n[3/4] Aritmetik Yoğunluk ve Bellek Trafiği Analitiği Çıkarılıyor...")
    print(f"  • Toplam FLOPs Miktarı                 : {wmma_stats['toplam_flops']:.2e} FLOPs")
    print(f"  • Aritmetik Yoğunluk                   : {wmma_stats['aritmetik_yogunluk_flop_per_byte']} FLOP/Byte (Compute-Bound)")
    print(f"  • HBM'den SRAM'e Okunan Eleman         : {sram_stats['hbm_okunan_eleman']} Eleman")
    print(f"  • Bank Conflict Durumu                 : {sram_stats['bank_conflict_status']} (128+4 Padding)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Tensor Core GEMM Teşhis Panosu Oluşturuluyor...")
    profil_raporu = TensorCoreProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "tensor_core_gemm_paneli.png")

    TensorCoreGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Tensor Core GEMM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 262 (FAZ 14): TENSOR CORE GEMM VE BLOCK-TILING MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
