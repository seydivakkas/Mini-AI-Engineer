"""
Day 314: Game-Theoretic Mechanism Design & Nash Bargaining
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

from src.oyun_teorisi_mekanizma import (
    MechanismConfig,
    MechanismResult,
    GameTheoreticEngine
)
from src.oyun_teorisi_profilleyici import GameTheoreticProfiler
from src.gorsellestirici import GameTheoreticGorsellestirici


def main():
    print("=" * 80)
    print("⚖️ DAY 314: OYUN TEORİK MEKANİZMA TASARIMI VE ÇOKLU AJAN NASH PAZARLIĞI")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Ajan Kümesi Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Çoklu Ajan Kümesi ve Değerleme Matrisleri Kuruluyor...")
    
    config = MechanismConfig(
        num_agents=4,
        num_goods_or_outcomes=5,
        total_compute_resource=100.0,
        seed=42
    )
    
    engine = GameTheoreticEngine(config)
    
    print(f"  • Ajan Sayısı: {config.num_agents}")
    print(f"  • Ayrık Karar / Çıktı Sayısı: {config.num_goods_or_outcomes}")
    print(f"  • Paylaştırılacak Toplam Hesaplama Kapasitesi: {config.total_compute_resource:.1f} TFLOPS")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: VCG ve Nash Pazarlığı Çözümü
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] VCG Dışsallık Ödemeleri ve Nash Pazarlık Dengesi Çözülüyor...")
    result: MechanismResult = engine.run_simulation()
    
    print(f"  ✓ VCG Optimal Çıktı: #{result.vcg_optimal_outcome}")
    print(f"  ✓ Toplam Sosyal Refah: {result.vcg_social_welfare:.2f}")
    print(f"  ✓ DSIC Dürüstlük Stratejik Kazancı: +{result.truthful_vs_manipulated_utility_gain:.4f}")
    print(f"  ✓ Toplam Nash Çarpımı (Surplus Product): {result.total_nash_product:.4f}")
    print(f"  ✓ Pareto Etkinliği: %{result.pareto_efficiency_pct:.2f}")
    
    print("\n  🔍 VCG Ödeme ve Net Fayda Dağılımı:")
    for a, pay in result.vcg_payments.items():
        u = result.vcg_net_utilities[a]
        print(f"    • {a:<12} -> Ödeme (Vergi): {pay:>6.2f} | Net Fayda: {u:>6.2f}")
        
    print("\n  🔍 Nash Pazarlığı Kaynak Dağılımı (TFLOPS):")
    for a, alloc in result.nash_bargaining_allocations.items():
        surp = result.nash_net_surpluses[a]
        print(f"    • {a:<12} -> Ayrılan Kaynak: {alloc:>6.2f} TFLOPS | Artık Rant: +{surp:>5.2f}")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = GameTheoreticProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "oyun_teorisi_paneli.png")
    GameTheoreticGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 OYUN TEORİSİ & NASH PAZARLIK TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<34}: {v}")
    print("=" * 80)
    print("✅ GÜN 314 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
