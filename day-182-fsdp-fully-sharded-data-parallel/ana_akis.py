"""
Day 182: Fully Sharded Data Parallel (FSDP) Ana Çalıştırma ve Doğrulama Akışı.
"""

import os
import sys
import torch
import torch.nn as nn

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.fsdp_sharding_motoru import ShardingLevel, FSDPKatmanSarmalayici
from src.fsdp_dagitik_yonetici import FSDPModelYoneticisi, FSDPBellekAnalizcisi
from src.gorsellestirici import FSDPGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 182 (FAZ 10): FULLY SHARDED DATA PARALLEL (FSDP) & ZERO-3 SHARDING ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: FSDP Katman Sharding & All-Gather Doğrulaması
    # -------------------------------------------------------------
    print("\n[1/4] FSDP Katman Sharding ve Bellek Boşaltma (Drop) Test Ediliyor...")
    world_size = 4
    rank = 0

    linear_layer = nn.Linear(512, 1024, bias=True)
    fsdp_layer = FSDPKatmanSarmalayici(
        module=linear_layer,
        world_size=world_size,
        rank=rank,
        sharding_level=ShardingLevel.FULL_SHARD,
    )

    stats = fsdp_layer.get_memory_stats()
    print(f"  • Toplam Parametre Sayısı   : {stats['toplam_parametre_numel']:,} ({stats['tam_model_bellek_mb']} MB)")
    print(f"  • Rank 0 Shard Parametresi  : {stats['shard_parametre_numel']:,} ({stats['shard_bellek_mb']} MB)")
    print(f"  • VRAM Tasarruf Oranı       : {stats['vram_tasarruf_orani']}")

    # -------------------------------------------------------------
    # ADIM 2: Çok Katmanlı FSDP Model Yöneticisi & İleri Geçiş
    # -------------------------------------------------------------
    print("\n[2/4] Çok Katmanlı FSDP Modeli Oluşturuluyor ve İleri Geçiş Gerçekleştiriliyor...")
    layers = [
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
    ]

    model_yonetici = FSDPModelYoneticisi(
        layers=layers,
        world_size=world_size,
        rank=rank,
        sharding_level=ShardingLevel.FULL_SHARD,
    )

    rapor = model_yonetici.get_toplam_bellek_raporu()
    print(f"  • Toplam Model Parametre    : {rapor['toplam_model_parametre']:,} ({rapor['tam_model_vram_mb']} MB)")
    print(f"  • Rank Başına Shard Bellek  : {rapor['fsdp_rank_vram_mb']} MB ({rapor['vram_tasarruf_kati']})")

    # Giriş tensörü ile ileri geçiş
    dummy_input = torch.randn(8, 256)
    cikis = model_yonetici.ileri_gecis(dummy_input)
    print(f"  ✓ FSDP İleri Geçiş Başarılı! Çıktı Şekli: {list(cikis.shape)}")

    # -------------------------------------------------------------
    # ADIM 3: Devasa LLM'ler İçin DDP vs ZeRO-1/2/3 Bellek Kıyaslama Tablosu
    # -------------------------------------------------------------
    print("\n[3/4] 64 GPU Kümesinde Büyük Dil Modelleri (LLM) İçin Bellek Karşılaştırma Analizi...")
    karsilastirma_tablosu = FSDPBellekAnalizcisi.buyuk_model_karsilastirma_tablosu(world_size=64)

    print("-" * 110)
    print(f"{'Model Adı':<15} | {'Param (B)':<10} | {'DDP VRAM':<12} | {'ZeRO-1 (Opt)':<14} | {'ZeRO-2 (Grad)':<14} | {'FSDP (ZeRO-3)':<14} | {'Tasarruf'}")
    print("-" * 110)
    for m in karsilastirma_tablosu:
        print(
            f"{m['model_adi']:<15} | {m['model_param_b']:<10.1f} | "
            f"{m['ddp_gb']:>8.1f} GB | {m['zero1_gb']:>10.1f} GB | "
            f"{m['zero2_gb']:>10.1f} GB | {m['fsdp_gb']:>10.1f} GB | "
            f"%{m['fsdp_vram_tasarrufu_yuzde']:.1f}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli FSDP Teşhis Panosu Oluşturuluyor...")
    layer_stats_list = [l.get_memory_stats() for l in model_yonetici.fsdp_layers if l.total_param_numel > 0]
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "fsdp_fully_sharded_data_parallel_paneli.png")

    FSDPGorsellestirici.fsdp_teshis_paneli_olustur(
        bellek_karsilastirma=karsilastirma_tablosu,
        layer_stats=layer_stats_list,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ FSDP Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 182: FULLY SHARDED DATA PARALLEL (FSDP) BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
