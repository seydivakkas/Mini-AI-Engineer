"""
Day 266 (FAZ 14): Apple Silicon Metal (MPS) & Metal Performance Shaders Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.apple_metal_motoru import (
    AppleSiliconUMAManager,
    MetalPerformanceShadersEngine,
)
from src.apple_metal_profilleyici import AppleMetalMPSProfilleyici
from src.gorsellestirici import AppleMetalGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 266 (FAZ 14): APPLE SILICON METAL (MPS) & METAL PERFORMANCE SHADERS MAC GPU OPTİMİZASYONU")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Birleşik Bellek (UMA) ve Sıfır Kopyalama Tahsisi
    # -------------------------------------------------------------
    print("\n[1/4] Apple Silicon UMA (Birleşik Bellek) Tensörü Tahsis Ediliyor...")
    uma_mgr = AppleSiliconUMAManager(memory_gb=128, bandwidth_gb_s=800.0)
    x = uma_mgr.allocate_shared_tensor("input_tokens", shape=(1, 128, 128), dtype=np.float32)
    np.random.seed(42)
    x[:] = np.random.randn(1, 128, 128).astype(np.float32)

    transfer_stats = AppleSiliconUMAManager.compare_transfer_overhead(x)
    print(f"  • Tensör Bellek Boyutu (1 x 128 x 128) : {transfer_stats['veri_boyutu_mb']} MB")
    print(f"  • PCIe 4.0 Host-to-Device Gecikmesi    : {transfer_stats['pcie_transfer_ms']} ms")
    print(f"  • Apple Silicon UMA Transfer Gecikmesi  : {transfer_stats['uma_transfer_ms']} ms (SIFIR KOPYALAMA)")

    # -------------------------------------------------------------
    # ADIM 2: MPS Graph Kaynaşık Transformer Bloğunun Yürütülmesi
    # -------------------------------------------------------------
    print("\n[2/4] MPS Graph Kaynaşık (RMSNorm + RoPE + SwiGLU GEMM) Çekirdeği Yürütülüyor...")
    norm_w = np.ones(128, dtype=np.float32)
    w_gate = np.random.randn(128, 256).astype(np.float32)
    w_up = np.random.randn(128, 256).astype(np.float32)
    w_down = np.random.randn(256, 128).astype(np.float32)

    out_mps, stats_mps = MetalPerformanceShadersEngine.execute_mps_fused_transformer_block(
        x, norm_w, w_gate, w_up, w_down
    )

    print(f"  • Metal Komut Kodlayıcı Sayısı (Encoder): {stats_mps['metal_command_encoders']} (Tek Kaynaşık Akış)")
    print(f"  • Çıktı Tensör Boyutu                  : {out_mps.shape}")
    print(f"  • SIMD-group Matris İşlem Mimarisi     : {stats_mps['simdgroup_matrix_boyutu']}")

    # -------------------------------------------------------------
    # ADIM 3: Sayısal Kararlılık ve Enerji Başarım Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Metal MPS ve UMA Enerji Verimliliği Analizi...")
    profil_raporu = AppleMetalMPSProfilleyici.basarim_profili_cikar()
    karsilastirma = profil_raporu["karsilastirma"]

    print(f"  • Llama-3-70B Çıkarım Hızı (Tok/s)     : {karsilastirma['cikarim_hizi_tok_s']['Apple_Metal_MPS']} tok/s (11x Hızlı)")
    print(f"  • 1000 Token Başına Enerji Tüketimi    : {karsilastirma['enerji_tuketimi_joule_1k_tok']['Apple_Metal_MPS']} J (8.6x Tasarruf)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Apple Silicon Metal MPS Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "apple_metal_mps_paneli.png")

    AppleMetalGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Apple Metal MPS Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 266 (FAZ 14): APPLE SILICON METAL MPS GPU OPTİMİZASYON MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
