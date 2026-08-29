"""
Day 316: Adversarial Byzantine Fault Tolerance & Robust Aggregation
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.bizans_hata_toleransi import (
    ByzantineSwarmConfig,
    ByzantineBenchmarkResult,
    ByzantineDefenseEngine
)
from src.bizans_profilleyici import ByzantineDefenseProfiler
from src.gorsellestirici import ByzantineDefenseGorsellestirici


def main():
    print("=" * 80)
    print("🛡️⚔️ DAY 316: BİZANS HATA TOLERANSI VE DÜŞMANCA SALDIRILARA KARŞI SAĞLAM TOPLAMA")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Dağıtık Sürü Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Dağıtık Ajan Sürüsü ve Bizans Saldırgan Düğümleri Kuruluyor...")
    
    config = ByzantineSwarmConfig(
        num_nodes=15,
        num_byzantine=4,
        param_dim=50,
        attack_type="sign_flipping",
        iterations=40,
        learning_rate=0.05,
        seed=42
    )
    
    engine = ByzantineDefenseEngine(config)
    
    print(f"  • Toplam Sürü Düğüm Sayısı (M): {config.num_nodes}")
    print(f"  • Düşmanca Bizans Düğüm Sayısı (f): {config.num_byzantine} (f < M/3)")
    print(f"  • Gradyan Parametre Boyutu (D): {config.param_dim}")
    print(f"  • Saldırı Tipi: {config.attack_type}")
    print(f"  • Saldırgan Düğüm İndeksleri: {engine.attacker_indices}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Dağıtık Optimizasyon ve Sağlam Toplama Kıyaslaması
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Naive Mean, Median, Trimmed Mean, Krum ve Bulyan Kıyaslanıyor...")
    result: ByzantineBenchmarkResult = engine.run_defense_benchmark()
    
    print(f"  ✓ Saldırı Azaltma Oranı (Mitigation Ratio): %{result.attack_mitigation_ratio_pct:.2f}")
    print(f"  ✓ Bizans Düğüm Tespit Kesinliği (Precision): %{result.byzantine_detection_precision_pct:.2f}")
    print(f"  ✓ Bizans Düğüm Tespit Duyarlılığı (Recall): %{result.byzantine_detection_recall_pct:.2f}")
    
    print("\n  🔍 Toplayıcılar Arası Ortalama Gradyan Uyumu (Cosine Fidelity):")
    for agg, cos in result.mean_cosine_fidelity.items():
        loss = result.final_objective_loss[agg]
        print(f"    • {agg:<14} -> Kosinüs: {cos:>7.4f} | Son Kayıp: {loss:>8.4f}")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = ByzantineDefenseProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "bizans_tolerans_paneli.png")
    ByzantineDefenseGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 BİZANS HATA TOLERANSI SÜRÜ TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<38}: {v}")
    print("=" * 80)
    print("✅ GÜN 316 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
