"""
Day 103: Multi-Head Latent Attention (MLA - DeepSeek V2/V3) Ana Akışı.
Sıkıştırılmış KV Latent, Ayrık RoPE, 32-katman LLM simülasyonu ve 6 panelli teşhis panosu.
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

from src.karsilastirma_laboratuvari import MLALaboratuvari
from src.gorsellestirici import MLAGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 103: DeepSeek Multi-Head Latent Attention (MLA) & Sıkıştırılmış KV Önbelleği")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: MLA Laboratuvarının Başlatılması
    # -------------------------------------------------------------
    print("\n[1/3] MLA Laboratuvarı Başlatılıyor (32 Katman, 16 Kafa, d_c=128, d_R=32)...")
    lab = MLALaboratuvari(
        dim=512,
        num_heads=16,
        head_dim=32,
        kv_latent_dim=128,
        q_latent_dim=256,
        rope_dim=32,
        katman_sayisi=32,
        cihaz=cihaz,
    )

    # -------------------------------------------------------------
    # ADIM 2: KV Cache Bellek Analizi ve Çıkarım Benchmark'ı
    # -------------------------------------------------------------
    dizi_uzunluklari = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    print(f"\n[2/3] KV Cache Bellek Analizi Yapılıyor (Batch=16, Bağlam: 512 -> 32,768 Token)...")
    bellek_raporu = lab.kv_cache_bellek_karsilastirmasi(batch_size=16, dizi_uzunluklari=dizi_uzunluklari)

    print("\n--- 32 KATMANLI MODEL İÇİN KV CACHE BELLEK AYAK İZİ (MB) ---")
    print(f"{'MİMARİ TÜRÜ':<22} | {'512 Tok':<9} | {'2048 Tok':<10} | {'8192 Tok':<10} | {'16384 Tok':<10} | {'32768 Tok':<10}")
    print("-" * 95)
    for m in ["MHA (16 KV Kafa)", "GQA (4 KV Kafa)", "DeepSeek MLA"]:
        degerler = bellek_raporu[m]
        print(f"{m:<22} | {degerler[0]:>7.1f}MB | {degerler[2]:>8.1f}MB | {degerler[4]:>8.1f}MB | {degerler[5]:>8.1f}MB | {degerler[6]:>8.1f}MB")
    print("-" * 95)

    print("\n[>>] Çıkarım Gecikmesi ve Throughput Ölçülüyor (Batch=8, SeqLen=512)...")
    gecikme_raporu = lab.gecikme_ve_throughput_olc(batch_size=8, seq_len=512, iterasyon=40)

    print("\n" + "=" * 95)
    print(f"{'MİMARİ VARYANTI':<25} | {'P50 (ms)':<10} | {'P90 (ms)':<10} | {'THROUGHPUT':<16} | {'PARAMETRE':<10}")
    print("-" * 95)
    for isim, met in gecikme_raporu.items():
        print(
            f"{isim:<25} | "
            f"{met['p50_ms']:>8.2f} ms | "
            f"{met['p90_ms']:>8.2f} ms | "
            f"{met['throughput_tps']:>11.1f} Tok/s | "
            f"{met['parametre_sayisi']:>10,}"
        )
    print("=" * 95)

    # 32,768 Bağlam Tasarruf Oranları
    mha_32k = bellek_raporu["MHA (16 KV Kafa)"][-1]
    gqa_32k = bellek_raporu["GQA (4 KV Kafa)"][-1]
    mla_32k = bellek_raporu["DeepSeek MLA"][-1]

    tasarruf_mha = ((mha_32k - mla_32k) / mha_32k) * 100.0
    tasarruf_gqa = ((gqa_32k - mla_32k) / gqa_32k) * 100.0

    print("\n[-] DEEPSEEK MLA MİMARİ ANALİZ RAPORU:")
    print(f"  * MHA 32k Bağlam KV Cache Belleği : {mha_32k:.1f} MB ({mha_32k/1024.0:.2f} GB)")
    print(f"  * GQA 32k Bağlam KV Cache Belleği : {gqa_32k:.1f} MB ({gqa_32k/1024.0:.2f} GB)")
    print(f"  * MLA 32k Bağlam KV Cache Belleği : {mla_32k:.1f} MB ({mla_32k/1024.0:.2f} GB)")
    print(f"  * MLA vs MHA Bellek Tasarrufu     : %{tasarruf_mha:.1f} VRAM Tasarrufu!")
    print(f"  * MLA vs GQA-4 Bellek Tasarrufu   : %{tasarruf_gqa:.1f} Ekstra VRAM Tasarrufu!")
    print("  * Endüstri Kararı                 : DeepSeek-V2/V3/R1 modellerinin bellek devrimi onaylandı.")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosunun Oluşturulması
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli DeepSeek MLA Teşhis Panosu Çiziliyor...")
    gorsellestirici = MLAGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "deepseek_mla_teshis_paneli.png",
    )
    gorsellestirici.pano_olustur(
        gecikme_raporu,
        bellek_raporu,
        dizi_uzunluklari=dizi_uzunluklari,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 103: DeepSeek Multi-Head Latent Attention Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
