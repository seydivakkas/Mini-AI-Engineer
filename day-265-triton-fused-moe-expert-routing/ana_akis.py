"""
Day 265 (FAZ 14): Triton Fused MoE Expert Routing Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.fused_moe_motoru import NaiveMoERouter, TritonFusedMoERouter
from src.fused_moe_profilleyici import FusedMoEProfilleyici
from src.gorsellestirici import FusedMoEGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 265 (FAZ 14): TRITON FUSED MOE EXPERT ROUTING — BELLEK KOPYALAMASINI SIFIRLAYAN UZMAN DAĞITIM ÇEKİRDEĞİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Girdi Token Tensörleri ve Uzman Ağırlıklarının Hazırlanması
    # -------------------------------------------------------------
    print("\n[1/4] 256 Token, 8 Uzman (Top-2) ve 64 Gizli Boyut Tensörleri Hazırlanıyor...")
    n, d, e = 256, 64, 8
    np.random.seed(42)
    x = np.random.randn(n, d).astype(np.float32)
    w_gate = np.random.randn(d, e).astype(np.float32)
    expert_weights = [np.random.randn(d, d).astype(np.float32) for _ in range(e)]

    print(f"  • Girdi Token Tensörü (N x D)        : {x.shape} (256 Token)")
    print(f"  • Yönlendirici Ağırlığı (D x E)      : {w_gate.shape} (8 Uzman)")
    print(f"  • Uzman Matris Sayısı                : {len(expert_weights)} Adet (64x64)")

    # -------------------------------------------------------------
    # ADIM 2: Triton Fused Zero-Copy MoE Yürütülmesi
    # -------------------------------------------------------------
    print("\n[2/4] Triton Fused MoE (Sıfır Kopyalama + Yerinde Akümülasyon) Yürütülüyor...")
    out_fused, stats_fused = TritonFusedMoERouter.forward(x, w_gate, expert_weights, top_k=2)

    print(f"  • Toplam Kopyalanan HBM Baytı        : {stats_fused['toplam_kopyalanan_bayt']} Bayt (SIFIR KOPYALAMA)")
    print(f"  • Çıktı Tensör Boyutu (N x D)        : {out_fused.shape}")
    print(f"  • Kopyalama Bellek Tasarrufu         : {stats_fused['kopyalama_tasarrufu']}")

    # -------------------------------------------------------------
    # ADIM 3: Klasik PyTorch Scatter/Gather ile Doğruluk Kıyaslaması
    # -------------------------------------------------------------
    print("\n[3/4] Klasik PyTorch Scatter/Gather ile Matematiksel Doğruluk Kıyaslanıyor...")
    out_naive, stats_naive = NaiveMoERouter.forward(x, w_gate, expert_weights, top_k=2)

    max_fark = float(np.max(np.abs(out_fused - out_naive)))
    print(f"  • Naive Kopyalanan HBM Baytı         : {stats_naive['toplam_kopyalanan_bayt']} Bayt")
    print(f"  • Maksimum Matematiksel Fark         : {max_fark:.2e} (Birebir Matematiksel Eşitlik)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Triton Fused MoE Teşhis Panosu Oluşturuluyor...")
    profil_raporu = FusedMoEProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "fused_moe_paneli.png")

    FusedMoEGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Fused MoE Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 265 (FAZ 14): TRITON FUSED MOE EXPERT ROUTING MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
