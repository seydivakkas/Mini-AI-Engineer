"""
Day 105: Sliding Window Attention (SWA - Mistral) & Rolling Buffer Cache Ana Akışı.
Sabit bellek karmaşıklığı O(W), dairesel tampon analizi ve 6 panelli teşhis panosu.
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

from src.swa_laboratuvari import SWALaboratuvari
from src.gorsellestirici import SWAGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 105: Mistral Sliding Window Attention (SWA) & Rolling Buffer Cache")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: SWA Laboratuvarının Başlatılması
    # -------------------------------------------------------------
    print("\n[1/3] SWA Laboratuvarı Başlatılıyor (32 Katman, 8 Q-Kafa, 2 KV-Kafa, Pencere W=512)...")
    lab = SWALaboratuvari(
        dim=512,
        num_q_heads=8,
        num_kv_heads=2,
        window_size=512,
        katman_sayisi=32,
        cihaz=cihaz,
    )

    # -------------------------------------------------------------
    # ADIM 2: KV Cache Bellek Analizi ve Alıcı Alan Hesabı
    # -------------------------------------------------------------
    dizi_uzunluklari = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    print(f"\n[2/3] KV Cache Bellek Analizi Yapılıyor (Batch=16, Bağlam: 512 -> 32,768 Token)...")
    bellek_raporu = lab.kv_cache_bellek_karsilastirmasi(batch_size=16, dizi_uzunluklari=dizi_uzunluklari)

    print("\n--- 32 KATMANLI MODEL İÇİN KV CACHE BELLEK AYAK İZİ (MB) ---")
    print(f"{'MİMARİ TÜRÜ':<32} | {'512 Tok':<9} | {'2048 Tok':<10} | {'8192 Tok':<10} | {'16384 Tok':<10} | {'32768 Tok':<10}")
    print("-" * 95)
    for isim, degerler in bellek_raporu.items():
        print(f"{isim:<32} | {degerler[0]:>7.1f}MB | {degerler[2]:>8.1f}MB | {degerler[4]:>8.1f}MB | {degerler[5]:>8.1f}MB | {degerler[6]:>8.1f}MB")
    print("-" * 95)

    alici_alan = lab.etkin_alici_alan_hesabi()
    print(f"\n[+] ETKİN ALICI ALAN (RECEPTIVE FIELD) ANALİZİ:")
    print(f"  * Pencere Boyutu (W)             : {alici_alan['pencere_boyutu']} Token")
    print(f"  * Toplam Katman Sayısı (L)       : {alici_alan['toplam_katman']}")
    print(f"  * Maksimum Alıcı Alan (L x W)    : {alici_alan['maksimum_alici_alan']:,} Token (Dolaylı Bağlam)")

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

    # 32k Bağlam Tasarruf Oranları
    keys = list(bellek_raporu.keys())
    full_32k = bellek_raporu[keys[0]][-1]
    swa_32k = bellek_raporu[keys[1]][-1]
    tasarruf = ((full_32k - swa_32k) / full_32k) * 100.0

    print("\n[-] MISTRAL SWA MİMARİ ANALİZ RAPORU:")
    print(f"  * Full Causal Attention 32k KV Cache : {full_32k:.1f} MB ({full_32k/1024.0:.2f} GB)")
    print(f"  * Mistral SWA (Rolling Cache) 32k    : {swa_32k:.1f} MB ({swa_32k/1024.0:.3f} GB - SABİT!)")
    print(f"  * VRAM Tasarruf Oranı                : %{tasarruf:.1f} Tasarruf!")
    print("  * Endüstri Kararı                    : Mistral 7B ve Mixtral 8x7B modellerinin mimari standardı onaylandı.")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosunun Oluşturulması
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Mistral SWA Teşhis Panosu Çiziliyor...")
    gorsellestirici = SWAGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "swa_rolling_cache_paneli.png",
    )
    gorsellestirici.pano_olustur(
        gecikme_raporu,
        bellek_raporu,
        alici_alan,
        dizi_uzunluklari=dizi_uzunluklari,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 105: Sliding Window Attention Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
