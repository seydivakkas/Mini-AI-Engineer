"""
Day 319: Free Energy Principle & Continuous Policy Iteration
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

from src.serbest_enerji_aktif_cikarim import (
    FEPConfig,
    FEPSimulationResult,
    GenerativeEnvironment,
    ActiveInferenceAgent
)
from src.serbest_enerji_profilleyici import FEPProfiler
from src.gorsellestirici import FEPGorsellestirici


def main():
    print("=" * 80)
    print("⚡🌐 DAY 319: SERBEST ENERJİ PRENSİBİ İLE SÜREKLİ POLİTİKA İTERASYONU (FEP & ACTIVE INFERENCE)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Üretici Ortam Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Karl Friston Üretici Modeli ve Aktif Çıkarım Ajanı Kuruluyor...")
    
    config = FEPConfig(
        num_states=4,
        num_obs=4,
        num_actions=3,
        horizon=10,
        precision_gamma=8.0,
        epistemic_weight=1.0,
        pragmatic_weight=1.0,
        seed=42
    )
    
    env = GenerativeEnvironment(true_reward_target=2, seed=42)
    agent = ActiveInferenceAgent(config)
    
    print(f"  • Gizli Durum Sayısı (S): {config.num_states} | Gözlem Sayısı (O): {config.num_obs}")
    print(f"  • Eylem Sayısı (A): {config.num_actions} (0: İpucu/Keşfet, 1: Hedef A, 2: Hedef B)")
    print(f"  • Politika Hassasiyeti (Gamma): {config.precision_gamma}")
    print(f"  • Gerçek Hedef Konumu: Hedef A (Durum 2)")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Aktif Çıkarım ve Politika İterasyon Döngüsü
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Algı-Eylem Döngüsü ve Beklenen Serbest Enerji G(pi) Minimizasyonu Yürütülüyor...")
    result: FEPSimulationResult = agent.run_active_inference_loop(env)
    
    print(f"  ✓ Hedefe Ulaşıldı mı: {'EVET' if result.goal_reached else 'HAYIR'}")
    print(f"  ✓ Toplam Epistemik Merak / Bilgi Kazanımı: {result.total_epistemic_gain:.4f} nats")
    print(f"  ✓ Son Varyasyonel Serbest Enerji (F): {result.final_vfe:.4f}")
    
    print("\n  🔍 Ajanın Durum ve Eylem İzi (Trajectory Trace):")
    for t, (s, a) in enumerate(zip(result.trajectory_states, result.trajectory_actions)):
        g_val = result.expected_free_energy_history[t]
        e_val = result.epistemic_value_history[t]
        p_val = result.pragmatic_value_history[t]
        print(f"    • Zaman t={t}: Durum s={s} -> Seçilen Eylem a={a} | G(pi): {g_val:>6.2f} (Epistemik: {e_val:>5.2f}, Pragmatik: {p_val:>5.2f})")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = FEPProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "serbest_enerji_paneli.png")
    FEPGorsellestirici.ciz(result, agent, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 SERBEST ENERJİ PRENSİBİ VE AKTİF ÇIKARIM TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<38}: {v}")
    print("=" * 80)
    print("✅ GÜN 319 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
