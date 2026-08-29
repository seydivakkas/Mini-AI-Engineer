"""
Day 102: Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) Ana Akışı.
KV Cache bellek optimizasyonu, 32-katman LLM simülasyonu ve 6 panelli teşhis panosu.
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

from src.karsilastirma_motoru import GQALaboratuvari
from src.gorsellestirici import GQAGorsellestirici


def main():
    print("=" * 90)
    print(">>> Day 102: Grouped-Query Attention (GQA) & KV Cache Optimizasyonu")
    print("=" * 90)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: GQA Laboratuvarının Başlatılması
    # -------------------------------------------------------------
    print("\n[1/3] GQA Laboratuvarı Başlatılıyor (32 Katman, 32 Query Başlığı, D=512)...")
    lab = GQALaboratuvari(dim=512, num_q_heads=32, katman_sayisi=32, cihaz=cihaz)

    # -------------------------------------------------------------
    # ADIM 2: KV Cache Bellek Tüketimi ve Çıkarım Benchmark'ı
    # -------------------------------------------------------------
    dizi_uzunluklari = [512, 1024, 2048, 4096, 8192]
    print(f"\n[2/3] KV Cache Bellek Analizi Yapılıyor (Batch=16, Bağlam: {dizi_uzunluklari})...")
    bellek_raporu = lab.kv_cache_bellek_analizi(batch_size=16, dizi_uzunluklari=dizi_uzunluklari)

    print("\n--- 32 KATMANLI MODEL İÇİN KV CACHE BELLEK AYAK İZİ (MB) ---")
    print(f"{'MİMARİ TÜRÜ':<30} | {'512 Tok':<10} | {'1024 Tok':<10} | {'2048 Tok':<10} | {'4096 Tok':<10} | {'8192 Tok':<10}")
    print("-" * 90)
    for m in ["MHA", "GQA", "MQA"]:
        degerler = bellek_raporu[m]
        print(f"{m:<30} | {degerler[0]:>8.1f} MB | {degerler[1]:>8.1f} MB | {degerler[2]:>8.1f} MB | {degerler[3]:>8.1f} MB | {degerler[4]:>8.1f} MB")
    print("-" * 90)

    print("\n[>>] Çıkarım Gecikmesi ve Throughput Ölçülüyor (Batch=8, SeqLen=512)...")
    gecikme_raporu = lab.gecikme_ve_throughput_olc(batch_size=8, seq_len=512, iterasyon=40)

    print("\n" + "=" * 90)
    print(f"{'MİMARİ VARYANTI':<32} | {'P50 (ms)':<10} | {'P90 (ms)':<10} | {'THROUGHPUT':<14} | {'PARAMETRE':<10}")
    print("-" * 90)
    for isim, met in gecikme_raporu.items():
        print(
            f"{isim:<32} | "
            f"{met['p50_ms']:>8.2f} ms | "
            f"{met['p90_ms']:>8.2f} ms | "
            f"{met['throughput_tps']:>9.1f} Tok/s | "
            f"{met['parametre_sayisi']:>10,}"
        )
    print("=" * 90)

    # 4096 Bağlam Tasarruf Oranı
    mha_4k = bellek_raporu["MHA"][3]
    gqa_4k = bellek_raporu["GQA"][3]
    mqa_4k = bellek_raporu["MQA"][3]
    gqa_tasarruf = ((mha_4k - gqa_4k) / mha_4k) * 100.0

    print("\n[-] GQA MİMARİ KARAR VE ANALİZ RAPORU:")
    print(f"  * MHA 4K Bağlam KV Cache Belleği : {mha_4k:.1f} MB")
    print(f"  * GQA 4K Bağlam KV Cache Belleği : {gqa_4k:.1f} MB (%{gqa_tasarruf:.1f} VRAM Tasarrufu!)")
    print(f"  * MQA 4K Bağlam KV Cache Belleği : {mqa_4k:.1f} MB")
    print("  * Endüstri Kararı                : LLaMA-3, Mistral-7B ve Gemma standardı olarak GQA-8 onaylandı.")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosunun Oluşturulması
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli GQA & KV Cache Teşhis Panosu Çiziliyor...")
    gorsellestirici = GQAGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "gqa_mqa_kv_cache_paneli.png",
    )
    gorsellestirici.pano_olustur(
        gecikme_raporu,
        bellek_raporu,
        dizi_uzunluklari=dizi_uzunluklari,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 90)
    print("[OK] Day 102: Grouped-Query Attention Analizleri Başarıyla Tamamlandı!")
    print("=" * 90)


if __name__ == "__main__":
    main()
