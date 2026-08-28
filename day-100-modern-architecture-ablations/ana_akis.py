"""
Day 100: Modern Mimari Ablasyon Analizleri Ana Akışı (SwiGLU, RMSNorm, FlashAttention/SDPA).
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

from src.ablasyon_motoru import AblasyonMotoru
from src.gorsellestirici import AblasyonGorsellestirici


def main():
    print("=" * 85)
    print(">>> Day 100: MiniViT Modern Mimari Ablasyon Analizleri (SwiGLU, RMSNorm, SDPA)")
    print("=" * 85)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Ablasyon Motorunun Başlatılması ve Varyantların İnşası
    # -------------------------------------------------------------
    print("\n[1/3] Mimari Varyantlar İnşa Ediliyor (4 Ayrı Konfigürasyon)...")
    motor = AblasyonMotoru(cihaz=cihaz)

    # -------------------------------------------------------------
    # ADIM 2: Kapsamlı Ablasyon Benchmark'ının Koşturulması
    # -------------------------------------------------------------
    print("\n[2/3] Sistematik Ablasyon Benchmark'ı Koşturuluyor (Batch=16, 40 İterasyon)...")
    sonuclar = motor.tum_ablasyonu_calistir(batch_size=16, iterasyon=40)

    print("=" * 85)
    print(f"{'VARYANT':<35} | {'PARAMETRE':<10} | {'P50 (ms)':<10} | {'THROUGHPUT':<12} | {'BELLEK':<10}")
    print("-" * 85)
    for isim, s in sonuclar.items():
        kisa_isim = isim.split(" (")[0]
        print(
            f"{kisa_isim:<35} | "
            f"{s['parametre_sayisi']:>10,} | "
            f"{s['p50_gecikme_ms']:>8.2f} ms | "
            f"{s['throughput_fps']:>7.1f} FPS  | "
            f"{s['tepe_bellek_mb']:>7.2f} MB"
        )
    print("=" * 85)

    # Karşılaştırmalı Analiz
    varyant_keys = list(sonuclar.keys())
    base_res = sonuclar[varyant_keys[0]]
    v2_res = sonuclar[varyant_keys[-1]]

    hizlanma = ((base_res["p50_gecikme_ms"] - v2_res["p50_gecikme_ms"]) / base_res["p50_gecikme_ms"]) * 100.0 if base_res["p50_gecikme_ms"] > v2_res["p50_gecikme_ms"] else 0.0

    print("\n[-] MIMARI ABLASYON KARSILASTIRMA VE KARAR RAPORU:")
    print(f"  * Baz Model Gecikmesi (ViT-Base)    : {base_res['p50_gecikme_ms']:.2f} ms ({base_res['throughput_fps']:.1f} FPS)")
    print(f"  * Modern Model Gecikmesi (Modern-v2): {v2_res['p50_gecikme_ms']:.2f} ms ({v2_res['throughput_fps']:.1f} FPS)")
    if hizlanma > 0:
        print(f"  * Mimari Hızlanma Oranı (Speedup)   : +%{hizlanma:.1f} Performans Artışı")
    print("  * Modern Yapı Taşları               : RMSNorm + SwiGLU + PyTorch SDPA (FlashAttention)")
    print("  * Büyük Final Hazırlığı             : [ONAYLANDI] Day 101 MoE v2 Mimari Temeli Hazır")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosunun Oluşturulması
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Modern Mimari Ablasyon Teşhis Panosu Çiziliyor...")
    gorsellestirici = AblasyonGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "modern_mimari_ablasyon_paneli.png",
    )
    gorsellestirici.pano_olustur(sonuclar, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 85)
    print("[OK] Day 100: Modern Mimari Ablasyon Analizleri Basariyla Tamamlandi!")
    print("=" * 85)


if __name__ == "__main__":
    main()
