"""
Day 183: DeepSpeed ZeRO-1/2/3 ve CPU/NVMe Bellek Boşaltma Ana Çalıştırma Akışı.
"""

import os
import sys
import torch
import torch.nn as nn

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.zero_offload_motoru import OffloadDevice, ZeROOffloadYapilandirma, CPUAdamWOptimizer
from src.zero_infinity_yonetici import ZeROInfinityKatmanSarmalayici, ZeROOffloadProfilleyici
from src.gorsellestirici import ZeROGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 183 (FAZ 10): DEEPSPEED ZeRO-1/2/3 & CPU/NVMe OFFLOAD (ZeRO-INFINITY) ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Host CPU AdamW Optimizer Motoru Doğrulaması
    # -------------------------------------------------------------
    print("\n[1/4] Host CPU AdamW Optimizer (12 Bayt/Param VRAM Boşaltma) Test Ediliyor...")
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
    )

    optimizer = CPUAdamWOptimizer(
        params=list(model.parameters()),
        lr=0.01,
        weight_decay=0.01,
    )

    # İleri ve Geri Geçiş
    dummy_x = torch.randn(16, 512)
    dummy_target = torch.randn(16, 256)
    cikis = model(dummy_x)
    loss = nn.functional.mse_loss(cikis, dummy_target)
    loss.backward()

    # CPU AdamW Güncellemesi (Device-to-Host -> CPU Compute -> Host-to-Device)
    optimizer.step()
    optimizer.zero_grad()

    opt_stats = optimizer.get_offload_memory_stats()
    print(f"  • Toplam Parametre Sayısı   : {opt_stats['toplam_parametre_sayisi']:,}")
    print(f"  • GPU VRAM Tasarrufu        : {opt_stats['gpu_vram_tasarrufu_mb']} MB ({opt_stats['vram_optimizer_azalmasi']})")
    print(f"  • Host CPU RAM Kullanımı    : {opt_stats['cpu_ram_tuketimi_mb']} MB")
    print(f"  ✓ CPU AdamW Güncelleme Adımı Başarıyla Tamamlandı! Kayıp: {loss.item():.4f}")

    # -------------------------------------------------------------
    # ADIM 2: ZeRO-Infinity Katman Sarmalama & Dinamik Boşaltma
    # -------------------------------------------------------------
    print("\n[2/4] ZeRO-Infinity Katman Sarmalama ve On-Demand PCIe Yükleme Test Ediliyor...")
    linear_layer = nn.Linear(512, 512)
    infinity_layer = ZeROInfinityKatmanSarmalayici(
        module=linear_layer,
        offload_device=OffloadDevice.CPU,
        compute_device="cpu",
    )

    out_inf = infinity_layer(dummy_x)
    print(f"  ✓ ZeRO-Infinity İleri Geçiş Başarılı! Çıktı Şekli: {list(out_inf.shape)}")

    # -------------------------------------------------------------
    # ADIM 3: 7B'den 1 Trilyona (1T) Kadar Model Bellek Profillemesi
    # -------------------------------------------------------------
    print("\n[3/4] Model Ölçeklerine Göre (7B - 1T) GPU VRAM, CPU RAM ve NVMe SSD Profillemesi...")
    profil_raporu = ZeROOffloadProfilleyici.coklu_model_profil_raporu()

    print("-" * 110)
    print(f"{'Model Adı':<18} | {'Param (B)':<10} | {'DDP GPU':<12} | {'Offload GPU':<14} | {'Offload CPU':<14} | {'Infinity NVMe':<14} | {'VRAM Tasarruf'}")
    print("-" * 110)
    for m in profil_raporu:
        print(
            f"{m['model_adi']:<18} | {m['model_param_b']:<10.1f} | "
            f"{m['ddp_gpu_vram_gb']:>8.1f} GB | {m['zero_offload_gpu_gb']:>10.1f} GB | "
            f"{m['zero_offload_cpu_gb']:>10.1f} GB | {m['zero_infinity_nvme_gb']:>10.1f} GB | "
            f"%{m['offload_vram_tasarrufu_yuzde']:.1f}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli ZeRO-Offload & ZeRO-Infinity Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "deepspeed_zero123_offload_paneli.png")

    ZeROGorsellestirici.zero_offload_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ ZeRO-Offload Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 183: DEEPSPEED ZeRO-OFFLOAD & ZeRO-INFINITY BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
