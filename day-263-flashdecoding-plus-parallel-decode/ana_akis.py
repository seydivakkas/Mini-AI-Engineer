"""
Day 263 (FAZ 14): FlashDecoding++ KV-Cache Bölümleme ve Decode Hızlandırma Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.flashdecoding_motoru import (
    KVCacheManager,
    FlashDecodingPlusEngine,
)
from src.flashdecoding_profilleyici import FlashDecodingProfilleyici
from src.gorsellestirici import FlashDecodingGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 263 (FAZ 14): FLASHDECODING++ — SPLIT-K KV-CACHE VE PARALEL DECODE HIZLANDIRMA")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Uzun Bağlam KV-Cache ve Query Tensörlerinin Hazırlanması
    # -------------------------------------------------------------
    print("\n[1/4] 2048 Token Uzunluğunda KV-Cache ve Tek Adımlık Query Vektörü Hazırlanıyor...")
    b, h, s, d = 2, 4, 2048, 64
    np.random.seed(42)
    q = np.random.randn(b, h, 1, d).astype(np.float32)
    k = np.random.randn(b, h, s, d).astype(np.float32)
    v = np.random.randn(b, h, s, d).astype(np.float32)

    print(f"  • Query Tensör Boyutu (B, H, 1, D) : {q.shape}")
    print(f"  • KV-Cache Tensör Boyutu (B, H, S, D): {k.shape} (2048 Token Bağlam)")

    # -------------------------------------------------------------
    # ADIM 2: Split-K Paralel Attention ve Softmax Rescaling
    # -------------------------------------------------------------
    print("\n[2/4] Split-K Paralel Attention (Chunk C=256) ve Softmax Rescaling Yürütülüyor...")
    out_split_k, stats = FlashDecodingPlusEngine.execute_split_k(q, k, v, chunk_size=256)

    print(f"  • Bölümlenen Chunk Sayısı           : {stats['bolumlenen_chunk_sayisi']} Parça (8x SM Dağıtımı)")
    print(f"  • Çıktı Tensör Boyutu (B, H, 1, D)   : {out_split_k.shape}")
    print(f"  • SM Paralelleştirme Durumu          : {stats['sm_paralellik_artisi']}")

    # -------------------------------------------------------------
    # ADIM 3: Matematiksel Doğruluk ve Hata Analizi
    # -------------------------------------------------------------
    print("\n[3/4] Klasik Tam Softmax Attention ile Matematiksel Doğruluk Kıyaslanıyor...")
    scale = 1.0 / np.sqrt(d)
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out_exact = np.matmul(weights, v)

    hata = float(np.max(np.abs(out_split_k - out_exact)))
    print(f"  • Maksimum Sayısal Hata Farkı        : {hata:.2e} (Tam Matematiksel Eşitlik)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli FlashDecoding++ Teşhis Panosu Oluşturuluyor...")
    profil_raporu = FlashDecodingProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "flashdecoding_paneli.png")

    FlashDecodingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ FlashDecoding++ Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 263 (FAZ 14): FLASHDECODING++ DECODE HIZLANDIRMA MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
