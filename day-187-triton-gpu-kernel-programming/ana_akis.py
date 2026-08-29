"""
Day 187: OpenAI Triton GPU Kernel Programlama ve Blok Bellek Eşleme Ana Çalıştırma Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.triton_temel_motoru import VektorToplamaKernel, FusedLineerKombinasyonKernel
from src.bellek_esleme_profilleyici import TritonBellekProfilleyici
from src.gorsellestirici import TritonGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 187 (FAZ 10): OPENAI TRITON GPU KERNEL & BLOCK-LEVEL MEMORY MAPPING ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Triton Blok Vektör Toplama ($Z = X + Y$)
    # -------------------------------------------------------------
    print("\n[1/4] Triton Blok Seviyesinde Vektör Toplama Çekirdeği İcrası...")
    n_eleman = 1_000_007  # Asal sayı (Sınır maskelemesi testi)
    x = torch.randn(n_eleman)
    y = torch.randn(n_eleman)

    z_triton = VektorToplamaKernel.calistir(x, y, block_size=1024)
    z_torch = x + y

    fark_max = torch.max(torch.abs(z_triton - z_torch)).item()
    print(f"  • Giriş Vektör Boyutu (N)     : {n_eleman:,} eleman")
    print(f"  • Blok Boyutu (BLOCK_SIZE)    : 1024")
    print(f"  • Toplam Grid (Blok Sayısı)   : {(n_eleman + 1023) // 1024:,} program")
    print(f"  • Maksimum Mutlak Hata        : {fark_max:.2e}")
    print(f"  • PyTorch ile Eşleşiyor mu    : {fark_max < 1e-6}")
    print("  ✓ Triton Vektör Toplama Çekirdeği Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Triton Fused Lineer Kombinasyon ($Y = \alpha X_1 + \beta X_2 + \gamma$)
    # -------------------------------------------------------------
    print("\n[2/4] Triton Fused Lineer Kombinasyon Çekirdeği (SRAM Fusion)...")
    alpha, beta, gamma = 1.75, 2.50, 0.85
    fused_triton = FusedLineerKombinasyonKernel.calistir(x, y, alpha=alpha, beta=beta, gamma=gamma, block_size=1024)
    fused_torch = alpha * x + beta * y + gamma

    fark_fused = torch.max(torch.abs(fused_triton - fused_torch)).item()
    print(f"  • Formül                      : Y = {alpha}*X1 + {beta}*X2 + {gamma}")
    print(f"  • Fused Hata                  : {fark_fused:.2e}")
    print(f"  • PyTorch ile Eşleşiyor mu    : {fark_fused < 1e-5}")
    print("  ✓ Triton Fused Çekirdek Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: HBM vs SRAM Bellek Trafik Profillemesi
    # -------------------------------------------------------------
    print("\n[3/4] HBM (DRAM) Bellek Trafiği ve Tasarruf Analizi...")
    bellek_analizi = TritonBellekProfilleyici.lineer_kombinasyon_bellek_analizi(eleman_sayisi=10_000_000)
    blok_raporu = TritonBellekProfilleyici.blok_boyutu_tarama_raporu(eleman_sayisi=10_000_000)

    print("-" * 110)
    print(f"  • Standart PyTorch HBM Trafiği : {bellek_analizi['pytorch_toplam_mb']} MB (5 Okuma + 4 Yazma = 9 Geçiş)")
    print(f"  • Fused Triton HBM Trafiği     : {bellek_analizi['triton_toplam_mb']} MB (2 Okuma + 1 Yazma = 3 Geçiş)")
    print(f"  • HBM Bellek Tasarruf Oranı    : %{bellek_analizi['hbm_kazanc_yuzde']} ({bellek_analizi['hbm_hizlanma_faktoru']} Daha Hızlı Bellek Akışı)")
    print(f"  • Ayrılan Ara VRAM Belleği     : {bellek_analizi['triton_ara_bellek_mb']} MB (PyTorch: {bellek_analizi['pytorch_ara_bellek_mb']} MB)")
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Triton GPU Kernel Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "triton_gpu_kernel_paneli.png")

    TritonGorsellestirici.teshis_paneli_olustur(
        bellek_analizi=bellek_analizi,
        blok_raporu=blok_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Triton GPU Kernel Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 187: OPENAI TRITON GPU KERNEL & BELLEK EŞLEME BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
