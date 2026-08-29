"""
Day 312: Infinite Context Streaming Recurrent Memory
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import torch
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.sonsuz_bellek_motoru import (
    StreamingMemoryConfig,
    StreamingMemoryResult,
    InfiniteContextStreamingEngine
)
from src.sonsuz_bellek_profilleyici import StreamingMemoryProfiler
from src.gorsellestirici import StreamingMemoryGorsellestirici


def main():
    print("=" * 80)
    print("🧠 DAY 312: SONSUZ BAĞLAM AKIŞI — SIKIŞTIRILMIŞ ÖZYİNELEMELİ AJAN BELLEĞİ")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Bellek Hücresi Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] O(1) Sabit Bellekli State-Space Özyinelemeli Hücre Kuruluyor...")
    
    config = StreamingMemoryConfig(
        d_model=32,
        d_state=32,
        context_stream_length=2000,
        decay_rate=0.998,
        num_needles=5,
        seed=42
    )
    
    engine = InfiniteContextStreamingEngine(config)
    
    print(f"  • Token Temsil Boyutu (d_model): {config.d_model}D")
    print(f"  • Özyinelemeli Durum Boyutu (d_state): {config.d_state}D")
    print(f"  • Akış Bağlam Uzunluğu: {config.context_stream_length:,} Token")
    print(f"  • Bellek Tutma Katsayısı (Lambda): {config.decay_rate}")
    print(f"  • Test Edilecek İğne (Needle) Sayısı: {config.num_needles}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: 2000 Token Akışı ve NIAH Geri Çağırma Testi
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] 2,000 Token Kesintisiz Akıtılıyor ve İğneler Sorgulanıyor...")
    result: StreamingMemoryResult = engine.run_streaming_benchmark()
    
    print(f"  ✓ NIAH Geri Çağırma Doğruluğu: %{result.retrieval_accuracy_pct:.2f}")
    print(f"  ✓ Bağlam Korunum İndeksi: {result.context_retention_index:.4f}")
    print(f"  ✓ Bellek Sıkıştırma Oranı: %{result.memory_compression_ratio_pct:.2f}")
    print(f"  ✓ Adım Başı Sabit Gecikme: {result.avg_step_latency_ms:.4f} ms (O(1))")
    
    print("\n  🔍 İğne Geri Çağırma Detayları:")
    for n in result.needle_results:
        status = "✅ GERİ ÇAĞRILDI" if n["is_recalled"] else "❌ KAYIP"
        print(f"    • İğne #{n['needle_id']} (Konum: {n['position']:>4}) -> Kosinüs: {n['cosine_similarity']:.3f} [{status}]")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = StreamingMemoryProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "sonsuz_bellek_paneli.png")
    StreamingMemoryGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 SONSUZ BAĞLAM SIKIŞTIRILMIŞ BELLEK TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<34}: {v}")
    print("=" * 80)
    print("✅ GÜN 312 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
