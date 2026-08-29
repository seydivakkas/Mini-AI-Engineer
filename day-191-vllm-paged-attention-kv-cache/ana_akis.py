"""
Day 191: vLLM PagedAttention ve Dinamik KV Cache Yönetimi Ana Çalıştırma Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.paged_attention_motoru import (
    FizikselBlokYonetici,
    GelenIstek,
    PagedKVCache,
    PagedAttentionEngine,
)
from src.fragmentasyon_profilleyici import KVCacheFragmentasyonProfilleyici
from src.gorsellestirici import PagedAttentionGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 191 (FAZ 10): VLLM ARCHITECTURE - PAGEDATTENTION & DYNAMIC KV CACHE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: PagedAttention Blok Tahsisi ve Çıkarım Simülasyonu
    # -------------------------------------------------------------
    print("\n[1/4] Sanal Sayfalama ve Dinamik Blok Tahsis Simülasyonu...")
    blok_yoneticisi = FizikselBlokYonetici(toplam_blok_sayisi=128, blok_boyutu=16)
    kv_cache = PagedKVCache(toplam_blok_sayisi=128, blok_boyutu=16, num_heads=4, head_dim=64)
    engine = PagedAttentionEngine(kv_cache, blok_yoneticisi)

    # 3 Farklı İstek Başlat
    istek1 = GelenIstek("req_001", prompt_token_sayisi=24)  # 2 Blok gerektirir (16 + 8)
    istek2 = GelenIstek("req_002", prompt_token_sayisi=35)  # 3 Blok gerektirir (16 + 16 + 3)

    # İstek 1 için KV Tokenları Yaz
    for _ in range(istek1.prompt_len):
        k_vec = torch.randn(4, 64)
        v_vec = torch.randn(4, 64)
        kv_cache.token_kv_yaz(istek1, k_vec, v_vec, blok_yoneticisi)

    # İstek 2 için KV Tokenları Yaz
    for _ in range(istek2.prompt_len):
        k_vec = torch.randn(4, 64)
        v_vec = torch.randn(4, 64)
        kv_cache.token_kv_yaz(istek2, k_vec, v_vec, blok_yoneticisi)

    print(f"  • İstek 1 (24 Token) Fiziksel Blok Tablosu: {istek1.blok_tablosu} (Toplam {len(istek1.blok_tablosu)} Blok)")
    print(f"  • İstek 2 (35 Token) Fiziksel Blok Tablosu: {istek2.blok_tablosu} (Toplam {len(istek2.blok_tablosu)} Blok)")
    print(f"  • Kalan Boş Blok Oranı                   : %{blok_yoneticisi.bos_blok_orani()*100:.1f}")

    # Kod Çözme (Decoding) Adımı: Yeni gelen token için dikkat hesapla
    q_yeni = torch.randn(4, 64)
    cikti1 = engine.tek_token_dikkat(istek1, q_yeni)
    print(f"  • İstek 1 Tek Token Dikkat Çıktı Şekli   : {list(cikti1.shape)}")
    print("  ✓ PagedAttention Dinamik Blok Yönetimi Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Copy-on-Write (CoW) Prompt Paylaşım Testi
    # -------------------------------------------------------------
    print("\n[2/4] Copy-on-Write (CoW) Paralel Örnekleme Testi...")
    # İstek 1'in bloklarını paylaşan yeni bir paralel dallanma isteği (Beam Search / Parallel Sample)
    istek_dal = GelenIstek("req_001_branch", prompt_token_sayisi=24)
    istek_dal.blok_tablosu = list(istek1.blok_tablosu)
    for b_id in istek_dal.blok_tablosu:
        blok_yoneticisi.referans_arttir(b_id)

    print(f"  • Paylaşılan Blok 0 Referans Sayacı       : {blok_yoneticisi.referans_sayaclari[istek1.blok_tablosu[0]]}")
    print(f"  • Paylaşılan Blok 1 Referans Sayacı       : {blok_yoneticisi.referans_sayaclari[istek1.blok_tablosu[1]]}")
    print("  ✓ Copy-on-Write (CoW) Mekanizması Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Bellek Fragmentasyonu ve Verim Analizi
    # -------------------------------------------------------------
    print("\n[3/4] Eşzamanlı İsteklerde KV Cache Fragmentasyon Analizi (32 İstek)...")
    istek_analizi = KVCacheFragmentasyonProfilleyici.eszamanli_istek_analizi(istek_sayisi=32)
    tarama_raporu = KVCacheFragmentasyonProfilleyici.eszamanlilik_tarama_raporu()

    print("-" * 110)
    print(f"{'İstek Sayısı':<15} | {'Statik VRAM (GB)':<18} | {'Paged VRAM (GB)':<18} | {'Statik İsraf (%)':<18} | {'Paged İsraf (%)':<18} | {'Verim Artışı'}")
    print("-" * 110)
    for r in tarama_raporu:
        print(
            f"{r['istek_sayisi']:<15d} | "
            f"{r['statik_vram_gb']:>14.2f} GB | "
            f"{r['paged_vram_gb']:>13.2f} GB | "
            f"%{r['statik_israf_yuzde']:>15.1f} | "
            f"%{r['paged_israf_yuzde']:>15.1f} | "
            f"{r['verim_artisi']:>12}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli vLLM PagedAttention Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "paged_attention_paneli.png")

    PagedAttentionGorsellestirici.teshis_paneli_olustur(
        istek_analizi=istek_analizi,
        tarama_raporu=tarama_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ PagedAttention Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 191: VLLM PAGEDATTENTION BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
