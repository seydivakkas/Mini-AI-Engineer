"""
Day 302: Recursive Meta-Architecture Search (DARTS + Bayesian Hypernet)
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.meta_nas_motoru import (
    MetaNASEngine,
    NASSearchConfig,
    NASSearchResult
)
from src.meta_nas_profilleyici import MetaNASProfiler
from src.gorsellestirici import MetaNASGorsellestirici


def main():
    print("=" * 80)
    print("👑 DAY 302: ÖZYİNELEMELİ META-MİMARİ ARAMA (DARTS & BAYESIAN HYPERNET)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Arama Uzayı ve Sentetik Veri Kümesi Hazırlığı
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Arama Uzayı ve Sentetik Görev Verisi Hazırlanıyor...")
    
    config = NASSearchConfig(
        num_nodes=3,
        channels=16,
        in_features=16,
        out_classes=4,
        num_epochs=25,
        lr_w=0.03,
        lr_alpha=0.005,
        tau_init=2.0,
        tau_min=0.25,
        seed=42
    )
    
    # Synthesize multi-class sequential classification task
    batch_size = 32
    seq_len = 12
    num_train_batches = 12
    num_val_batches = 6
    
    train_loader = []
    for _ in range(num_train_batches):
        x = torch.randn(batch_size, config.channels, seq_len)
        y = torch.randint(0, config.out_classes, (batch_size,))
        train_loader.append((x, y))
        
    val_loader = []
    for _ in range(num_val_batches):
        x = torch.randn(batch_size, config.channels, seq_len)
        y = torch.randint(0, config.out_classes, (batch_size,))
        val_loader.append((x, y))
        
    print(f"  • Arama Düğümleri (Nodes): {config.num_nodes}")
    print(f"  • Kanal Boyutu: {config.channels} | Sınıf Sayısı: {config.out_classes}")
    print(f"  • Eğitim Örnekleri: {batch_size * num_train_batches} | Doğrulama: {batch_size * num_val_batches}")
    print(f"  • Aday Operasyonlar: Identity, Zero, Conv3x3, Conv5x5, AvgPool, GELULinear")
    
    # -------------------------------------------------------------
    # ADIM 2: Bi-Level Meta-Mimari Arama Motorunu Başlatma
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2] Bi-Level Optimizasyon ve Bayesian Hypernet ile Arama Başlatılıyor...")
    engine = MetaNASEngine(config)
    search_result: NASSearchResult = engine.run_search(train_loader, val_loader)
    
    print(f"  ✓ Arama Tamamlandı! Süre: {search_result.search_time_sec:.2f} saniye")
    print(f"  ✓ Son Gumbel Sıcaklığı (τ): {search_result.final_tau:.3f}")
    print(f"  ✓ Süpernet Son Doğrulama Başarımı: %{search_result.search_history['val_acc'][-1]:.2f}")
    
    # -------------------------------------------------------------
    # ADIM 3: Aday Mimariler ve Pareto Optimum Sınırı
    # -------------------------------------------------------------
    print("\n🎯 [ADIM 3] Çok Amaçlı Pareto Optimum Sınırı Çıkarılıyor...")
    best = search_result.best_candidate
    pareto_list = search_result.pareto_frontier
    
    print(f"  • Değerlendirilen Toplam Aday: {len(search_result.all_candidates)}")
    print(f"  • Pareto Optimum Aday Sayısı: {len(pareto_list)}")
    print(f"  • 🏆 Seçilen En İyi Mimari (Gen): {best.gene}")
    print(f"    - Doğruluk: %{best.accuracy:.2f}")
    print(f"    - FLOPs: {best.flops_m:.4f} MFLOPs")
    print(f"    - Gecikme: {best.latency_ms:.2f} ms")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = MetaNASProfiler.profile_search(search_result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "meta_nas_paneli.png")
    MetaNASGorsellestirici.ciz(search_result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 META-NAS ARAMA VE PERFORMANS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<30}: {v}")
    print("=" * 80)
    print("✅ GÜN 302 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
